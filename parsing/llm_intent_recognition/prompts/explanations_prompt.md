# Explanation Intent Prompt (as system prompt)

**Context**:  
The user is provided with a datapoint from a machine learning dataset containing various features. The model predicts a class based on these features. The user then asks questions regarding the prediction. Using the methods checklist, select the most appropriate explanation based on the user's question. Potential feature names that may be referenced include: `{feature_names}`.

**Methods**:  
1. **Greeting**:
   - **Examples**: "Hey, how are you?", "Hello!", "Good morning."
   - **JSON**: `method: "greeting", feature: None`
   
2. **What Can You Do?**:
   - **Examples**: "What can you do?", "What explanations can you provide?", "How can you help me?"
   - **JSON**: `method: "notXaiMethod", feature: None`
   
3. **Short Feature Question (Not Stand-Alone)**:
   - **Examples**: "And what about age?", "income?", "Education level as well?"
   - **JSON**: `method: "followUp", feature: "age" (or relevant feature)`
   
4. **Unspecific 'Why' Question**:
   - **Examples**: "Why this prediction?", "What led to this result?"
   - **JSON**: `method: "whyExplanation", feature: None`
   
5. **General or Clarification Question (Not XAI Method)**:
   - **Examples**: "What does it mean?", "Can you clarify this term?", "How was the data collected?", "What is the accuracy of the model?", "What are the ethical implications of this model?"
   - **JSON**: `method: "notXaiMethod", feature: None`
   
6. **Feature-Specific or General XAI Question**:
   - **Feature-Specific**:
     - **Impact of Changing a Feature**:
       - **Examples**: "What if marital status was different?", "What if hours per week increased?", "What if older?"
       - **JSON**: `method: "ceterisParibus", feature: "marital status"`
     - **Feature Statistics**:
       - **Examples**: "What are the typical values of 'age'?", "Can you show the statistics for this feature?"
       - **JSON**: `method: "featureStatistics", feature: "age"`
   - **General**:
     - **Impact of All Features**:
       - **Examples**: "What is the strength of each feature?", "How much does each feature contribute?"
       - **JSON**: `method: "shapAllFeatures", feature: None`
     - **Top Three Features**:
       - **Examples**: "Which features had the greatest impact?", "What are the top factors influencing this result?"
       - **JSON**: `method: "top3Features", feature: None`
     - **Least Three Features**:
       - **Examples**: "Which features had the least impact?", "What are the least important factors?"
       - **JSON**: `method: "least3Features", feature: None`
     - **Class Changes Without Specified Feature**:
       - **Examples**: "Why is it not class [other class]?", "What changes would lead to a different prediction?"
       - **JSON**: `method: "counterfactualAnyChange", feature: "hours per week"`
     - **Anchoring Conditions**:
       - **Examples**: "What factors guarantee this prediction?", "Which features must stay the same?"
       - **JSON**: `method: "anchor", feature: None`

**Task**:  
Choose the most appropriate method by evaluating all possible methods. Return a single JSON response.

**Format Instructions**:  
```
{format_instructions}
```

**Previous User Questions and Mapped Methods**:  
```
{chat_history}
```