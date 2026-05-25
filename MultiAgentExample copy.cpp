#include "8BLlamaSingleAgent.hpp"

int main()
{
    std::vector<std::unordered_map<std::string, std::string>> cmnd(3);

    std::string context1 =
        "Transformers are deep learning models mainly used in natural language "
        "processing tasks such as text generation, translation, and question answering. "
        "The main idea behind transformers is the attention mechanism, which helps the "
        "model focus on important words in a sentence. Unlike RNNs, transformers process "
        "all words in parallel, making training much faster.";

    cmnd[0]["text"] = context1;
    cmnd[0]["header"] = "system";

    std::string question1_1 =
        "What is the main idea behind transformers?";

    std::string question1_2 =
        "Why are transformers faster than RNNs during training?";

    std::string context2 =
        "Diffusion models are generative AI models used to create images, audio, "
        "and videos. These models first add noise to data step by step and then "
        "learn how to remove the noise to recover meaningful data. Stable Diffusion "
        "is a popular example of a diffusion model used for image generation.";

    cmnd[1]["text"] = context2;
    cmnd[1]["header"] = "system";

    std::string question2_1 =
        "What do diffusion models learn during training?";

    std::string question2_2 =
        "Name one popular diffusion model used for image generation.";

    std::string context3 =
        "Reinforcement Learning is a machine learning method where an agent learns "
        "by interacting with an environment. The agent takes actions and receives "
        "rewards or penalties based on those actions. The goal of the agent is to "
        "maximize the total reward over time. Reinforcement Learning is widely used "
        "in robotics, games, and autonomous systems.";

    cmnd[2]["text"] = context3;
    cmnd[2]["header"] = "system";

    std::string question3_1 =
        "What is the goal of the agent in Reinforcement Learning?";

    std::string question3_2 =
        "Where is Reinforcement Learning commonly used?";

    std::unordered_map<std::string, std::string> question;
    question["text"] = question1_1;
    question["header"] = "user";

    MultiAgent agent(3, 2000, 1, cpu);
    agent.AddCommand(cmnd);

    agent.ask(question, 0);

    std::cout << "Context: " << cmnd[0]["text"] << std::endl;
    std::cout << "Question: " << question["text"] << std::endl;
    while (true)
    {
        std::string nextChar = agent.GetNext(0);
        if (nextChar == "endOfText")
        {
            std::cout << "\n\n[Finished]" << std::endl;
            break;
        }
        std::cout << nextChar << std::flush;
    }
    std::cout << std::endl;

    
    question["text"] = question2_1;

    agent.ask(question, 1);

    std::cout << "Context: " << cmnd[1]["text"] << std::endl;
    std::cout << "Question: " << question["text"] << std::endl;
    while (true)
    {
        std::string nextChar = agent.GetNext(1);
        if (nextChar == "endOfText")
        {
            std::cout << "\n\n[Finished]" << std::endl;
            break;
        }
        std::cout << nextChar << std::flush;
    }
    std::cout << std::endl;

    question["text"] = question3_1;
    agent.ask(question, 2);

    std::cout << "Context: " << cmnd[2]["text"] << std::endl;
    std::cout << "Question: " << question["text"] << std::endl;
    while (true)
    {
        std::string nextChar = agent.GetNext(2);
        if (nextChar == "endOfText")
        {
            std::cout << "\n\n[Finished]" << std::endl;
            break;
        }
        std::cout << nextChar << std::flush;
    }
    std::cout << std::endl;

    agent.reset(0);
    question["text"] = question1_2;
    agent.ask(question, 0);

    std::cout << "Context: " << cmnd[0]["text"] << std::endl;
    std::cout << "Question: " << question["text"] << std::endl;
    while (true)
    {
        std::string nextChar = agent.GetNext(0);
        if (nextChar == "endOfText")
        {
            std::cout << "\n\n[Finished]" << std::endl;
            break;
        }
        std::cout << nextChar << std::flush;
    }
    std::cout << std::endl;

    return 0;
}