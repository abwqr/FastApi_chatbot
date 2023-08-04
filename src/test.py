from chain import qa

print("Results:")
question = ["what is the code of conduct?", "Requirements for BS Cyber Security?"]

# result = qa({"question":question, "chat_history":chat_history})
for q in question:   
    result = qa({"query":q})
    # chat_history = [(question, result["answer"])]
    print(result['result'])