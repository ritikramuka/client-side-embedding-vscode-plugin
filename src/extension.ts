// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from "vscode";
import { pipeline } from "@xenova/transformers";

// Initialize the embedding model
let embeddingModel: any;

async function initializeEmbeddingModel() {
	embeddingModel = await pipeline("feature-extraction", "Xenova/gte-small");
}

// Generate embeddings for a given text
async function getEmbedding(text: string): Promise<number[]> {
	if (!embeddingModel) {
		await initializeEmbeddingModel();
	}
	const output = await embeddingModel(text, {
		pooling: "mean",
		normalize: true,
	});
	return Array.from(output.data);
}

// Pre-computed embeddings for OpenAPI Swagger documentation
interface OpenAPIChunk {
	text: string;
	embedding: number[];
}

let openAPIEmbeddings: OpenAPIChunk[] = [];

// Load pre-computed embeddings into memory (e.g., from IndexedDB or a local file)
async function loadOpenAPIEmbeddings() {
	// Example: Load embeddings from a JSON file
	const embeddingsFile = vscode.Uri.file("path/to/embeddings.json");
	const fileContent = await vscode.workspace.fs.readFile(embeddingsFile);
	openAPIEmbeddings = JSON.parse(fileContent.toString());
}

// Search for relevant chunks based on the user's query
async function searchRelevantChunks(
	query: string,
	topK: number = 3
): Promise<OpenAPIChunk[]> {
	const queryEmbedding = await getEmbedding(query);

	// Compute cosine similarity between the query and each chunk
	const similarities = openAPIEmbeddings.map((chunk) => ({
		similarity: cosineSimilarity(queryEmbedding, chunk.embedding),
		chunk,
	}));

	// Sort by similarity and return the top K results
	return similarities
		.sort((a, b) => b.similarity - a.similarity)
		.slice(0, topK)
		.map((result) => result.chunk);
}

// Format the context for GitHub Copilot Chat
function formatContext(chunks: OpenAPIChunk[]): string {
	return chunks.map((chunk) => chunk.text).join("\n\n");
}

// Pre-defined prompt for GitHub Copilot Chat
const predefinedPrompt = (context: string, userQuery: string): string => `
You are a helpful assistant for developers integrating an API. Below is some context from the API documentation:

---
${context}
---

If the context above is relevant to the user's query, use it to provide a detailed and accurate response. If the context is not relevant, omit it and respond based on your own understanding.

User Query: ${userQuery}
`;

// Utility function for cosine similarity
export function cosineSimilarity(vecA: number[], vecB: number[]): number {
	if (vecA.length !== vecB.length) {
		throw new Error("Vectors must have the same length");
	}
	const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
	const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
	const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
	return dotProduct / (magnitudeA * magnitudeB);
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log(
		'Congratulations, your extension "client-side-embedding-vscode-plugin" is now active!'
	);

	// Load OpenAPI embeddings when the extension is activated
	loadOpenAPIEmbeddings();

	// Register the chat participant
	vscode.chat.createChatParticipant("api-helper", async (request: vscode.ChatRequest) => {
		const query = request.prompt;
	
		// Search for relevant chunks
		const relevantChunks = await searchRelevantChunks(query);
	
		// Format the context
		const context = formatContext(relevantChunks);
	
		// Create the prompt for GitHub Copilot Chat
		const prompt = predefinedPrompt(context, query);
	
		try {
			const [model] = await vscode.lm.selectChatModels({ vendor: 'copilot', family: 'gpt-4o' });
			const response = await model.sendRequest(prompt, {}, request.token);
	
			// Return the response as a ChatResult
			return {
				content: response,
				// Add other properties as needed
			};
		} catch (err) {
			// Making the chat request might fail because
			// - model does not exist
			// - user consent not given
			// - quota limits were exceeded
			if (err instanceof vscode.LanguageModelError) {
				console.log(err.message, err.code, err.cause);
				if (err.cause instanceof Error && err.cause.message.includes('off_topic')) {
					return {
						content: vscode.l10n.t("I'm sorry, I can only explain computer science concepts."),
						// Add other properties as needed
					};
				}
			} else {
				// add other error handling logic
				throw err;
			}
		}
	});
}

// This method is called when your extension is deactivated
export function deactivate() { }
