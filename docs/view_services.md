## App Overview


### Full Flow 
The app is organized into **three main pipelines**, as shown below:

```mermaid
flowchart TD
    subgraph app_report["app/report/"]
        A1[Load Results] --> A2[Build LLM]
        A2 --> A3[Generate Report]
    end

    subgraph app_drift["app/drift/"]
        B1[Load Embeddings] --> B2[Embeddings Visualization]
        B2 --> B3[Detect Drift]
    end

    subgraph app_database["app/database/"]
        C1[Upload Data] --> C2[Load Data]
        C2 --> C3[Text Visualization]
        C3 --> C4[Embedding]
        C4 --> C5[Store in DB]
    end
```

##### 🔹 `app/database/`  
- Upload raw text data  
- Visualize text  
- Generate embeddings  
- Store embeddings to DB (Milvus / FAISS)

##### 🔹 `app/drift/`  
- Load stored embeddings  
- Visualize embedding space  
- Run drift detection (Evidently, Distance metrics)

##### 🔹 `app/report/`  
- Load drift results  
- Run report generation  
- Integrate with LLMs for explanation generation