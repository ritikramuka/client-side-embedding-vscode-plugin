import * as fs from "fs";
import { pipeline } from "@xenova/transformers";

async function generateEmbeddingsForOpenAPI() {
    const embeddingModel = await pipeline(
        "feature-extraction",
        "Xenova/gte-small"
    );

    // Parse your OpenAPI Swagger documentation into chunks
    const openAPIChunks = [
        {
            text: "Endpoint: GET /users\nDescription: Retrieve a list of users.",
        },
        {
            text: "Endpoint: POST /auth\nDescription: Authenticate with the API using an API key.",
        },
        // Add more chunks as needed
    ];

    // Generate embeddings for each chunk
    const embeddings = [];
    for (const chunk of openAPIChunks) {
        const embedding = await embeddingModel(chunk.text, {
            pooling: "mean",
            normalize: true,
        });
        embeddings.push({
            text: chunk.text,
            embedding: Array.from(embedding.data),
        });
    }

    // Save embeddings to a file
    fs.writeFileSync("embeddings.json", JSON.stringify(embeddings));
}

generateEmbeddingsForOpenAPI();
