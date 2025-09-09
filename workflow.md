# AI-Powered Personal Fitness Trainer for Sedentary lifestyle

- I want to create a distributed multi agent workflow for this that levarges LangGraph, Tools, Memory

## Purpose

- This is a Streamlit app that acts as an AI-powered personal health & fitness trainer.
- It generates personalized diet + workout plans, provides daily coaching chats, and lets users track their progress over time.

## Key components

1. AI Agents (Gemini models)

* Dietary Planner → Creates personalized diet plans.

* Fitness Trainer → Generates workout routines.

* Progress Tracker → Analyzes user progress and check-ins.

* Daily Coach → Conversational chatbot that answers questions and motivates.

* Team Lead → Combines diet + workout into a holistic strategy.

2. LangGraph Workflow

* retrieve_context → Gets user’s latest message + profile.

* generate_response → Calls daily_coach with:

* Context

* User profile

* Last few messages from memory

* Ensures conversations feel adaptive and continuous.

3. Conversation Memory

ConversationMemory class stores chat history in .pkl files.

Saves user ↔ coach messages with timestamps.

Keeps last 5–10 messages for continuity in conversation.

- persistant memory to store static informations 
1. user_name, user_BMI (ie. wieght hieght age), user_fitness_goals, user_diet preference, user_activity_level, PLAN_CREATED, chat_history

- non-persistant memory to store streaming chat

1. to save the session chat


## Requirements

- on home screen user will enter these variables

### First page 

- name, age, weight, height, fitness goals, activity level, diet preference [Basically user_name, user_BMI, goals]
- UserProfile class stores:

 [Name, age, weight, height, activity level, diet preference, fitness goal.]

- after getting these things in Input a new session_ID will be created for that user 
- have to apply a logic here that If this user_name already exist in the databse, take the existing session_ID (save in the persistant memory) [Decide the persistant memory here]

### second page 

- Now we will create Fitness plan here for the user calling the planner_agent
- If this PLAN_CREATED already in the persistant memory show this plan on the screen 
- else create the plan 
- working of the planner_agent It will use the user_BMI and Gemini_API and DuckDuckGo tool that will take the key information about the health from [WHO API], So after this user will be able to see the Fitness plan for the week attributes [Eating habbits, Exercises] for the week {PLAN_CREATED} save this created plan in the persistant memory

[Meal plans and workout plans.

Profiles are saved/loaded as JSON files (user_data/*.json).]


### third page 

* Daily Coach → Conversational chatbot that answers questions and motivates.

* PROMPTING OF THIS IS VERY IMPORTANT

- This will be a Conversational chatbot where user can talk about what changes it feels, what to be changed in the plans, how to improve in future
- This Chat bot will act a motivation person to the user that will keep him excited always be Funny
- This Chat bot will have the context about all the persistant memory of the user that is 

1. user_name 
2. user_BMI (ie. wieght hieght age)
3. user_fitness_goals
4. user_diet preference
5. user_activity_level
6. PLAN_CREATED
7. chat_history

- So accordingly this chatbot will talk in a adaptive way to the user

### fourth page 

- Progress Tracker 
- this will store the daily Input given by the user like what user did today in eating habits and exercide
- User can log new weight + notes.
- a section to store the daily weight 
- A dummy graph to show the PRogress for the week
- Analyzes user progress and check-ins.

### fifth page

- Team Lead → Combines diet + workout into a holistic strategy.
- Provide a summary of what happend for the meanwhile process the changes in the body
- user can upolad photo and a button to add the photo this will be DUMMY
- Progress Tracker generates a 📊 progress report (added to conversation + displayed).

### sixth page

- EXERCISE INSTRUCTOR
- use mediapie and blazepose API to track the body angles of the body and apply a logic to count the sit ups and some exersice that is possible 
- If the body posture is not perfect lines will trun red else green 










