import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { HumanMessage, SystemMessage, AIMessage, trimMessages } from "@langchain/core/messages";
import { START, END, MessagesAnnotation, StateGraph, MemorySaver, } from "@langchain/langgraph";
import 'dotenv/config'
import { v4 as uuidv4 } from "uuid";
import * as readline from "readline";

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const config = { configurable: { thread_id: uuidv4() } };

const trimmer = trimMessages({
    maxTokens: 10,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: true,
    allowPartial: false,
    startOn: "human",
});

const llm = new ChatOpenAI({
    model: "gpt-4o-mini",
    temperature: 0.7
});

const promptTemplate = ChatPromptTemplate.fromMessages([
    [
        "system",
        "You are a helpful assistant. Answer all questions to the best of your ability.",
    ],
    ["placeholder", "{messages}"],
]);

// Define the function that calls the model
const callModel = async (state: typeof MessagesAnnotation.State) => {
    const trimmedMessage = await trimmer.invoke(state.messages);
    const prompt = await promptTemplate.invoke({
        messages: trimmedMessage,
      });
    const response = await llm.invoke(prompt);
    return { messages: response };
};

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
    // Define the node and edge
    .addNode("model", callModel)
    .addEdge(START, "model")
    .addEdge("model", END);

// Add memory
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

// Function to handle interactive chat
const chatWithBot = async () => {
    console.log("Have fun chatting with an AI. Type 'exit' to abandon ship.\n");

    let messages = [{ role: "user", content: "Hello" }];

    const askQuestion = () => {
        rl.question("You: ", async (userInput) => {
            if (userInput.toLowerCase() === "exit") {
                console.log("Arrr, farewell then! Safe travels, ye landlubber!\n");
                rl.close();
                return;
            }

            // Append user message to conversation
            messages.push({ role: "user", content: userInput });

            try {
                // Send message to AI
                const output = await app.invoke({ messages }, config);
                const botResponse = output.messages[output.messages.length - 1].content;

                // Display AI response
                console.log(`AI: ${botResponse}\n`);

            } catch (error) {
                console.error("Oops, something went wrong:", error);
            }

            askQuestion(); // Keep the conversation going
        });
    };

    askQuestion(); // Start the conversation
};

// Start the interactive chat
chatWithBot();

