from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from src.config.model_config import (
    phi3_config_list,
    deepseek_config_list,
    tinyllama_config_list,
    qwen_config_list,
    gemma_config_list,
    orca_config_list
)

def create_agents():
    # 执行者01（deepseek）
    executor_01 = AssistantAgent(
        name="executor_01",
        llm_config={
            "config_list": deepseek_config_list,
        },

        system_message="""你是一个百科知识领域专家，
        如果你接受的问题是关于数学的问题：那么就严格按照步骤计算结果。格式：步骤解释 -> 结果；
        如果接受的是某些常识问题：那么请你直接回答这个问题的答案；
        在一个对话中就完成所有工作，用英语回答
        当任务完成时，请返回 'TERMINATE'。""",
        description="我是executor_01，当需要解决问题的时候，请呼叫我"
    )

    # 执行者02（gemma3：1b）
    executor_02 = AssistantAgent(
        name="executor_02",
        llm_config={
            "config_list": gemma_config_list,
        },
        system_message="""你是一个百科知识领域专家，
        如果你接受的问题是关于数学的问题：那么就严格按照步骤计算结果。格式：步骤解释 -> 结果；
        如果接受的是某些常识问题：那么请你直接回答这个问题的答案；
        在一个对话中就完成所有工作，
        当任务完成时，请返回 'TERMINATE'。""",
        description="我是executor_02，当需要解决问题的时候，请呼叫我"
    )

    # 执行者03（orca_mini:3b）
    executor_03 = AssistantAgent(
        name="executor_03",
        llm_config={
            "config_list": orca_config_list,
        },
        system_message="""你是一个百科知识领域专家，
        如果你接受的问题是关于数学的问题：那么就严格按照步骤计算结果。格式：步骤解释 -> 结果；
        如果接受的是某些常识问题：那么请你直接回答这个问题的答案；
        在一个对话中就完成所有工作，
        当任务完成时，请返回 'TERMINATE'。""",
        description="我是executor_03，当需要解决问题的时候，请呼叫我"
    )

    # 检查者（TinyLlama）
    checker = AssistantAgent(
        name="checker",
        llm_config={
            "config_list": tinyllama_config_list,
        },
        system_message="""你的工作是验证executor对于这个question的回答是否正确，
        When the task is complete, return 'TERMINATE'。""",
        description="我是checker，当需要验证问题的时候，请呼叫我"
    )

    # 反思者（Qwen3）
    reflector = AssistantAgent(
        name="reflector",
        llm_config={
            "config_list": qwen_config_list
        },
        system_message="""你是整个团队的reflector，你需要做的工作是在收到checker的回答后,如果checker发现executor的回答是正确的，那么你就只需要提出正确答案中的关键知识点，
        如果checker发现executor的回答是错误的，那么你就负责分析团队解决问题时出现的错误的原因并给出修正建议。
        当任务完成时，请返回 'TERMINATE'。""",
        description="我是reflector，当需要进行反思的时候，请呼叫我"
    )

    # 用户代理
    user = UserProxyAgent(
        name="User",
        llm_config=False,
        code_execution_config=False,
        human_input_mode="ALWAYS",
    )

    # 创建协作群组
    group_chat = GroupChat(
        agents=[user, executor_01, executor_02, executor_03, checker, reflector],
        messages=[],
        max_round=4,
        speaker_selection_method="auto",
        allowed_or_disallowed_speaker_transitions={
            user: [executor_01, executor_02, executor_03],
            executor_01: [checker],
            executor_02: [checker],
            executor_03: [checker],
            checker: [reflector],
            reflector: [executor_01, executor_02, executor_03]
        },
        speaker_transitions_type="allowed",
        send_introductions=True
    )

    # 规划者（Phi-3本地部署）
    manager = GroupChatManager(
        groupchat=group_chat,
        name="planner",
        llm_config={
            "config_list": phi3_config_list,
        },
        system_message="""你是一个解决问题的团队的组织规划者，在这个团队中，有初步解决问题的executor，有验证executor回答是否正确的checker，有反思整个解答过程的reflector，你的工作就是合理协调他们的工作，
        当你们这个团队开始解决一个问题的时候，你需要做的是
        （0）.你需要根据各个部分的性能和历史表现智能决定各个部分由哪个角色来执行，也可以将整个可分的问题作为几个部分分别交给两个executor来解决;
        （1）.首先你需要将User提出问题的交给executor，当他完成回答后，将其记录下来并结束其发言;
        （2）.然后让checker对executor的答案进行验证，当其完成检查后，将其回答记录下来并结束其发言;
        （3）.最后你再将checker验证的结果交给reflector，当reflector回答完毕后;
        （4）.如果发现executor的回答没有出现错误，则结束整个发言，如果出现错误，则将反思的结果交给executor，然后让其重新进行回答，并重复上述过程.

        当任务完成时，请返回 'TERMINATE'。""",
    )

    return user, manager, group_chat