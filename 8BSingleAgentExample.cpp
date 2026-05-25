#include "8BLlamaSingleAgent.hpp"

int main()
{
    std::unordered_map<std::string, std::string> cmnd;
    std::string context = "The Transformer model is a deep learning architecture mainly used for natural language processing tasks such as text generation, translation, and summarization. Unlike older recurrent neural networks (RNNs), Transformers process all words in parallel instead of one by one. This makes training much faster and allows the model to learn long-range relationships between words more effectively. The core idea behind Transformers is the attention mechanism. Attention helps the model focus on the most important words in a sentence when predicting the next word or understanding meaning. A special type called Self-Attention allows each word to interact with every other word in the same sentence. The Transformer architecture contains two major parts: the encoder and the decoder. The encoder reads and understands the input text, while the decoder generates the output text. Modern language models like GPT mainly use the decoder part, while models like BERT mainly use the encoder part. Inside the model, words are converted into vectors called embeddings. Positional encoding is added because the Transformer itself does not naturally understand word order. Multi-head attention is used so the model can learn different types of relationships at the same time. Transformers also use feedforward neural networks, layer normalization, and residual connections to improve training stability. Because of their scalability and parallel processing ability, Transformers became the foundation of modern large language models such as GPT, Llama, and Gemini.";    

    // std::string question = "Why is the attention mechanism important in Transformer models, and how does it help compared to older RNN-based models?";

    cmnd["text"] = context;
    cmnd["header"] = "system";
    llama agent(1, 2000, cmnd, cpu);    
    std::cout << "[*] Context: " << context << std::endl;

    while(true)
    {
        std::string question;
        std::cout << "[user]>> ";
        std::getline(std::cin, question);
        if(question == "exit")
        {
            std::cout << "[*] chal pheli fursat m nikal, meko kya bura lgta h" << std::endl;
            break;
        }
        cmnd["header"] = "user";
        cmnd["text"] = question;

        agent.ask(cmnd);
        std::cout << "[Machine]>> ";
        for (int i = 0; i < 1000; i++)
        {
            std::string text = agent.GetNext();
            if (text == "endOfText")
            {
                std::cout << "\n\n[Finished]" << std::endl;
                break;
            }
            std::cout << text << std::flush;
        }
        std::cout << "\n\n";
    }

    return 0;
}