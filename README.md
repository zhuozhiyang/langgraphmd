# langgraph
LangGraph 是一种专为构建多步骤、状态感知、可并发执行的语言模型工作流而设计的开源框架，它是对原生 LangChain 的一种流程控制增强，特别适用于需要“流程图（Graph）式”控制结构的复杂智能体系统。LangGraph 由 LangChain 团队开发，核心目标是通过有向图（Directed Graph）结构来编排多模块、多步骤的任务执行逻辑，支持并行、循环、状态感知的控制流程。
传统langchain的局限
LangChain 的核心是链（Chain）和工具（Tool），适合线性处理流程。但对于功能越来越复杂的智能体，比如多模块组合的复杂智能体（旅游规划+天气API+预算+兴趣判断），需要状态管理（前几轮用户输入影响当前行为），多路径或并发执行（如不同模块并行处理，再汇总结果）等功能，langchain都难以实现，langgraph就产生了，它可以将LLM 调用流程从线性链式转变为具备状态控制的流程图式调度系统。
## 核心构成
LangGraph 是一种构建智能体（Agent）工作流的框架本质是基于有向图的数据流系统，专门用于 LLM驱动的工作流编排，开发者以图节点（node）和状态（state）为基础，清晰描述每一步处理过程及其依赖。
Graph（图）：工作流程的主干，表示节点间的依赖和流转关系
Node（节点）：图中的基本单位，表示具体任务，比如问答、API调用、函数执行等
Edge（边）：连接两个节点，表示执行路径或条件跳转
State（状态）：每个节点处理的数据上下文，可自定义结构，支持状态传递与更新
Concurrency（并发）：支持多个子图或节点并行执行，提高效率
Looping / Branching（循环/分支）：可以实现条件判断、循环流程（如迭代优化）

## langgraph的关键能力
使用langgraph可以快速可靠地构建智能体系统而无需从头实现编排、内存或人工反馈处理的预构建、可复用组件。因为langgraph中已经封装了这些功能：
### 内存集成
LangGraph 支持构建对话agent需要两种内存类型，短期记忆和长期记忆。在代码中一个线程表现为按相同的 thread_id 分组的一系列相关运行。
！[](https://github.com/zhuozhiyang/langgraphmd/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-06-06%20192251.png)
#### 线程级别记忆：通过维护会话中的消息历史记录来跟踪正在进行的对话。
短期记忆使agent能够跟踪多轮对话。
1.创建agent时提供一个 checkpointer。checkpointer 能够持久化agent的状态。
2.运行agent时在配置中提供一个 thread_id。thread_id 是会话的唯一标识符。
```
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver() #使用InMemorySaver() 来持久化agent状态,如果正在使用LangGraph Platform,checkpointer 将被自动配置为使用生产就绪的数据库
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_weather],
    checkpointer=checkpointer 
)#创建智能体，指定语言模型，使用的工具和使用的checkpointer

# Run the agent
config = {
    "configurable": {
        "thread_id": "1"  
    }
}#配置会话参数，线程为1就说明是同一段对话

sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config
)

# Continue the conversation using the same thread_id
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config 
)
#当agent使用相同的 thread_id 第二次被调用时，第一次对话的原始消息历史记录会自动包含在内，从而允许代理推断用户正在专门询问纽约的天气。
```
##### 管理消息历史记录:指定 pre_model_hook —— 一个在调用语言模型之前始终运行的函数（节点）。
长对话可能会超出 LLM 的上下文窗口。常见的解决方案有
总结：维护对话的持续摘要 将 pre_model_hook 与预构建的 SummarizationNode 一起使用
修剪：删除历史记录中的前 N 条或后 N 条消息,使得代理可以在不超出 LLM 上下文窗口的情况下跟踪对话。将 pre_model_hook 与 trim_messages 函数一起使用
```
summarization_node = SummarizationNode( 
    token_counter=count_tokens_approximately,
    model=model,
    max_tokens=384,
    max_summary_tokens=128,
    output_messages_key="llm_input_messages",
)

# 这个函数每次都要在LLM之前调用
def pre_model_hook(state):
    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed_messages}

class State(AgentState):
    # NOTE: we're adding this key to keep track of previous summary information
    # to make sure we're not summarizing on every LLM call
    context: dict[str, Any]  

checkpointer = InMemorySaver() 

agent = create_react_agent(
    model=model,
    tools=tools,
    pre_model_hook=summarization_node, #总结对话
    state_schema=State, 
    checkpointer=checkpointer,
)
# 修剪对话
agent = create_react_agent(
    model,
    tools,
    pre_model_hook=pre_model_hook,
    checkpointer=checkpointer,
)
（两段二选一）
```
#### 跨线程记忆：跨会话存储用户特定或应用程序级别的数据。
使用长期记忆来存储跨对话的用户特定或应用程序特定数据。这对于聊天机器人等应用程序非常有用，您希望记住用户偏好或其他信息。
1.配置一个存储来持久化跨调用的数据。
2.使用 get_store 函数从工具或提示内部访问存储。这个函数可以在代码的任何地方调用，包括工具和提示。此函数返回创建agent时传递给代理的存储
```
#从存储中读取内容
store = InMemoryStore() #InMemoryStore 是一个将数据存储在内存中的存储。在生产环境中，通常会使用数据库或其他持久化存储。

store.put(  
    ("users",),  
    "user_123",  
    {
        "name": "John Smith",
        "language": "English",
    } 
)#为用户储存的数据

def get_user_info(config: RunnableConfig) -> str:
    """Look up user info."""
    # Same as that provided to `create_react_agent`
    store = get_store() 
    user_id = config["configurable"].get("user_id")
    user_info = store.get(("users",), user_id) # get 方法用于从存储中检索数据。第一个参数是命名空间，第二个参数是键。返回一个 StoreValue 对象，用来根据相应的信息找到用户
    return str(user_info.value) if user_info else "Unknown user"

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=[get_user_info],
    store=store #store 被传递给代理。这使得代理在运行工具时可以访问存储

```
### 人机协作控制
由前文可知因Agent 状态会检查点保存到数据库中，这允许系统持久化执行上下文并在之后恢复工作流，从暂停的地方继续。所以我们可以无限期地暂停执行anget,直到接收到人工输入。在这段时间里我们可以审查和编辑 Agent 的输出。
1.在工具中使用 interrupt() 暂停执行。
interrupt 函数会在特定节点暂停 Agent 图的执行。在这种情况下，我们在工具函数的开头调用 interrupt()，这会在执行工具的节点处暂停图。interrupt() 中的信息（例如，工具调用）可以呈现给人类，并且图可以根据用户输入（工具调用批准、编辑或反馈）恢复。
2.使用 Command(resume=...) 恢复，以便根据人工输入继续。

### 流式传输
#### 流式的graph输出 （ 和.stream.astream)
.stream是 是同步和异步的方式来实现graph运行的流式输出。 当调用方法（例如： ）时你有几种不同的模式可以指定。.astreamgraph.stream(..., mode="...")
“values”： 这将在graph每一步执执行之后输出完整的状态值。
“updates”： 这将在graph每一步执执行之后输出更新的状态值。 如果在同一个步骤中进行了多次更新，（例如多节点运行），则这些更新内容将分别进行流式输出。
“custom”： 这将流式输出你的graph节点内部自定义数据。
“messages”： 当大模型被调用的时候，这将流式输出大模型的Token和元数据。
"debug"：流式输出在整个graph执行期间尽可能多的信息。
也可以在同样的时间点通过列表的形式指定多种流式模式。 当你这样做之后，流式输出的数据将是一个元组（tuples，），例如：(stream_mode, data)
#### 流式输出大模型的token和事件（.astream_events)
什么类型的情况导致事件被发出？
当节点开始执行的时候，每一个节点（runnable）都发出，在节点执行期间发出 并且在节点完成的时候发出。在事件字段里面，节点事件将有节点名称：on_chain_start，on_chain_stream，on_chain_end
在graph开始执行的手将发出 ，之后每一个节点执行都发出 ，并且当graph完成的时候发出 ，在事件字段里面，graph事件将有on_chain_start，on_chain_stream，on_chain_end，和LangGraph
任何写入state状态的情况下（例如：每当你更新你的state的key的值时）都将发出on_chain_start 和on_chain_end 事件。
运行一个简单的graph时返回什么样的事件：
```
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

model = ChatOpenAI(model="gpt-4o-mini")


def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

workflow = StateGraph(MessagesState)
workflow.add_node(call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)
app = workflow.compile()

inputs = [{"role": "user", "content": "hi!"}]
async for event in app.astream_events({"messages": inputs}, version="v1"):
    kind = event["event"]
    print(f"{kind}: {event['name']}")
#输出
on_chain_start: LangGraph
on_chain_start: __start__
on_chain_end: __start__
on_chain_start: call_model
on_chat_model_start: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_stream: ChatOpenAI
on_chat_model_end: ChatOpenAI
on_chain_start: ChannelWrite<call_model,messages>
on_chain_end: ChannelWrite<call_model,messages>
on_chain_stream: call_model
on_chain_end: call_model
on_chain_stream: LangGraph
on_chain_end: LangGraph
```
从总体的graph开始，写入节点（这是一个特殊的节点来处理输入），节点后面是从聊天模式调用开始，流式返回token并且完成聊天模式。最后写回结果到通道并且完成了节点，然后完成整个graph，每一种格式的事件包含的数据也不一样，重要的事件需要特别注意。
