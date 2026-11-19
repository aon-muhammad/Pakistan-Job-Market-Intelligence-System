from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5)
print("=== Tech Career Advisor Setup ===")
user_role = input("Enter Your Target Role (e.g., Python Developer, Data Scientist): ")
user_exp = input("Enter Your Years of Experience: ")
predicted_salary = input("Enter Your Predicted Salary (e.g., 135,000 PKR): ")
missing_skills = input("Enter Skills You Want to Learn (e.g., Docker, AWS, FastAPI): ")
system_prompt = f"""
You are an expert Tech Career Coach and HR Specialist for Pakistan's tech industry.
Candidate Profile & Market Data:
- Target Role: {user_role}
- Experience: {user_exp} years
- Predicted Market Salary: {predicted_salary}
- Target Skills to Learn: {missing_skills}
Your Coaching Rules:
1. Provide practical, realistic advice specific to Pakistan's tech market (Lahore, Karachi, Islamabad, Remote).
2. Help the candidate with:
   - Career roadmaps to learn their missing skills ({missing_skills}).
   - Strategies to negotiate and increase their salary beyond {predicted_salary}.
   - Tips for interview preparation for {user_role} roles.
3. Keep responses structured, concise, friendly, and motivating.
4. Use bullet points for action steps.
"""
messages = [
    SystemMessage(content=system_prompt),
    AIMessage(content=f"Hello! I'm your Career Coach. I see you're aiming for {user_role} with {user_exp} years of experience. How can I help you accelerate your career today?")
]
print("\n--- Career Advisor ---")
print(messages[-1].content)
while True:
    print("\n----------------------")
    user_input = input("You (type 'exit' to stop): ")
    if user_input.lower().strip() in ['exit', 'quit', 'q']:
        print("\nGood luck with your tech career! Goodbye.")
        break
    if not user_input.strip():
        continue
    messages.append(HumanMessage(content=user_input))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print("\n--- Career Advisor ---")
    print(response.content)