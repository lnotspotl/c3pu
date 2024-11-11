Generates RAG file containing eviction data from policies.

Policies include:
    Task02 - Single linear layer architecture
    Task03 - MLP (2 layers) architecture
    Task04 - RNN architecture

Example run command: generates RAG file with 100 instruction steps for each policy.
python -m raggen \
    --output_folder="./outputs" \
    --rag_file="rag_data_100.txt" \
    --model_checkpoint=68000 \
    --num_instructions=100    