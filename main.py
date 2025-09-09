
"""AI Health & Fitness Plan Application.

This module implements a Streamlit-based web application for personalized health and fitness planning.
It provides features for creating customized diet and workout plans, tracking user progress,
and offering daily coaching through a conversational interface.

The application uses multiple AI agents powered by the Gemini model for generating plans and
coaching, integrates with DuckDuckGo for web searches, and manages user data with persistent storage.
"""

import os
import streamlit as st
import pickle
import json
import torch
import asyncio
import nest_asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import Graph
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# Application configuration
st.set_page_config(page_title="AI Health & Fitness Plan", page_icon="🏋️‍♂️", layout="wide")

# Load environment variables
load_dotenv()

# Fix asyncio and torch compatibility issues
nest_asyncio.apply()
torch._C._log_api_usage_once("my_app")
torch.classes.__path__ = []

# Initialize Google API Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Constants
USER_DATA_DIR = "user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)

# Initialize embedding model for vector storage
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize AI Agents
def _create_dietary_planner():
    """Creates a dietary planner agent for generating personalized meal plans."""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Creates personalized dietary plans based on user input.",
        instructions=[
            "Generate a diet plan with breakfast, lunch, dinner, and snacks.",
            "Consider dietary preferences like Keto, Vegetarian, or Low Carb.",
            "Ensure proper hydration and electrolyte balance.",
            "Provide nutritional breakdown including macronutrients and vitamins.",
            "Suggest meal preparation tips for easy implementation.",
            "If necessary, search the web using DuckDuckGo for additional information.",
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

def _create_fitness_trainer():
    """Creates a fitness trainer agent for generating customized workout routines."""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Generates customized workout routines based on fitness goals.",
        instructions=[
            "Create a workout plan including warm-ups, main exercises, and cool-downs.",
            "Adjust workouts based on fitness level: Beginner, Intermediate, Advanced.",
            "Consider weight loss, muscle gain, endurance, or flexibility goals.",
            "Provide safety tips and injury prevention advice.",
            "Suggest progress tracking methods for motivation.",
            "If necessary, search the web using DuckDuckGo for additional information.",
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

def _create_progress_tracker():
    """Creates a progress tracker agent for analyzing user progress."""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Tracks and analyzes user progress over time.",
        instructions=[
            "Analyze progress based on user input and historical data.",
            "Provide motivational feedback and suggestions for improvement.",
            "Compare current status with goals and previous check-ins.",
            "Suggest adjustments to diet or exercise if progress stalls.",
        ],
        markdown=True
    )

def _create_daily_coach():
    """Creates a daily coach agent for conversational interaction."""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Provides daily coaching and answers fitness/diet questions.",
        instructions=[
            "Engage in friendly, motivational conversation with the user.",
            "Answer questions about fitness, nutrition, and health.",
            "Provide daily tips based on the user's goals.",
            "Track daily check-ins and provide feedback.",
            "Maintain a positive and encouraging tone.",
            "Refer to historical data when appropriate.",
        ],
        tools=[DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

def _create_team_lead():
    """Creates a team lead agent to combine diet and workout plans."""
    return Agent(
        model=Gemini(id="gemini-2.0-flash-exp"),
        description="Combines diet and workout plans into a holistic health strategy.",
        instructions=[
            "Merge personalized diet and fitness plans for a comprehensive approach.",
            "Ensure alignment between diet and exercise for optimal results.",
            "Suggest lifestyle tips for motivation and consistency.",
            "Provide guidance on tracking progress and adjusting plans over time.",
            "Use tables for clear presentation when possible.",
        ],
        markdown=True
    )

dietary_planner = _create_dietary_planner()
fitness_trainer = _create_fitness_trainer()
progress_tracker = _create_progress_tracker()
daily_coach = _create_daily_coach()
team_lead = _create_team_lead()

class UserProfile:
    """Manages user profile data including personal details and progress tracking."""
    
    def __init__(self, name, age, weight, height, activity_level, dietary_preference, fitness_goal):
        """Initializes a user profile with personal and fitness information.
        
        Args:
            name (str): User's name.
            age (int): User's age in years.
            weight (float): User's weight in kilograms.
            height (float): User's height in centimeters.
            activity_level (str): User's activity level (e.g., Sedentary, Moderately Active).
            dietary_preference (str): User's dietary preference (e.g., Keto, Vegetarian).
            fitness_goal (str): User's fitness goal (e.g., Weight Loss, Muscle Gain).
        """
        self.name = name
        self.age = age
        self.weight = weight
        self.height = height
        self.activity_level = activity_level
        self.dietary_preference = dietary_preference
        self.fitness_goal = fitness_goal
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.check_ins = []
        self.meal_plans = []
        self.workout_plans = []
    
    def add_check_in(self, weight=None, notes=""):
        """Adds a progress check-in to the user's profile.
        
        Args:
            weight (float, optional): Current weight in kilograms.
            notes (str, optional): Additional notes for the check-in.
        """
        check_in = {
            "date": datetime.now(),
            "weight": weight if weight is not None else self.weight,
            "notes": notes
        }
        self.check_ins.append(check_in)
        self.updated_at = datetime.now()
        if weight is not None:
            self.weight = weight
    
    def add_meal_plan(self, plan):
        """Adds a meal plan to the user's profile.
        
        Args:
            plan (str): The meal plan content.
        """
        self.meal_plans.append({
            "date": datetime.now(),
            "plan": plan
        })
    
    def add_workout_plan(self, plan):
        """Adds a workout plan to the user's profile.
        
        Args:
            plan (str): The workout plan content.
        """
        self.workout_plans.append({
            "date": datetime.now(),
            "plan": plan
        })
    
    def to_dict(self):
        """Converts the user profile to a serializable dictionary.
        
        Returns:
            dict: A dictionary representation of the user profile.
        """
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        return {
            "name": self.name,
            "age": self.age,
            "weight": self.weight,
            "height": self.height,
            "activity_level": self.activity_level,
            "dietary_preference": self.dietary_preference,
            "fitness_goal": self.fitness_goal,
            "created_at": convert_datetime(self.created_at),
            "updated_at": convert_datetime(self.updated_at),
            "check_ins": [
                {
                    "date": convert_datetime(check_in["date"]),
                    "weight": check_in["weight"],
                    "notes": check_in["notes"]
                } for check_in in self.check_ins
            ],
            "meal_plans": [
                {
                    "date": convert_datetime(plan["date"]),
                    "plan": plan["plan"]
                } for plan in self.meal_plans
            ],
            "workout_plans": [
                {
                    "date": convert_datetime(plan["date"]),
                    "plan": plan["plan"]
                } for plan in self.workout_plans
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Creates a UserProfile instance from a dictionary.
        
        Args:
            data (dict): Dictionary containing user profile data.
        
        Returns:
            UserProfile: A new UserProfile instance.
        """
        def parse_datetime(dt_str):
            if isinstance(dt_str, str):
                return datetime.fromisoformat(dt_str)
            return dt_str
        
        user = cls(
            data["name"],
            data["age"],
            data["weight"],
            data["height"],
            data["activity_level"],
            data["dietary_preference"],
            data["fitness_goal"]
        )
        user.created_at = parse_datetime(data["created_at"])
        user.updated_at = parse_datetime(data["updated_at"])
        user.check_ins = [
            {
                "date": parse_datetime(check_in["date"]),
                "weight": check_in["weight"],
                "notes": check_in["notes"]
            } for check_in in data["check_ins"]
        ]
        user.meal_plans = [
            {
                "date": parse_datetime(plan["date"]),
                "plan": plan["plan"]
            } for plan in data["meal_plans"]
        ]
        user.workout_plans = [
            {
                "date": parse_datetime(plan["date"]),
                "plan": plan["plan"]
            } for plan in data["workout_plans"]
        ]
        return user

def save_user_profile(user):
    """Saves a user profile to a JSON file.
    
    Args:
        user (UserProfile): The user profile to save.
    """
    filename = f"{USER_DATA_DIR}/{user.name.lower().replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(user.to_dict(), f, default=str)

def load_user_profile(name):
    """Loads a user profile from a JSON file.
    
    Args:
        name (str): The name of the user.
    
    Returns:
        UserProfile or None: The loaded user profile or None if not found.
    """
    filename = f"{USER_DATA_DIR}/{name.lower().replace(' ', '_')}.json"
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return UserProfile.from_dict(data)
    except FileNotFoundError:
        return None

class ConversationMemory:
    """Manages conversation history for a user."""
    
    def __init__(self, user_id):
        """Initializes conversation memory for a user.
        
        Args:
            user_id (str): Unique identifier for the user.
        """
        self.user_id = user_id
        self.memory_file = f"{USER_DATA_DIR}/{user_id}_memory.pkl"
        self.messages = []
        self.load_memory()
    
    def add_message(self, role, content):
        """Adds a message to the conversation history.
        
        Args:
            role (str): The role of the message sender ('user' or 'assistant').
            content (str): The content of the message.
        """
        self.messages.append({
            "role": role, 
            "content": content, 
            "timestamp": datetime.now().isoformat()
        })
        self.save_memory()
    
    def get_recent_messages(self, n=5):
        """Retrieves the most recent messages from conversation history.
        
        Args:
            n (int): Number of recent messages to retrieve.
        
        Returns:
            list: List of recent messages.
        """
        return self.messages[-n:]
    
    def save_memory(self):
        """Saves conversation history to a pickle file."""
        with open(self.memory_file, "wb") as f:
            pickle.dump(self.messages, f)
    
    def load_memory(self):
        """Loads conversation history from a pickle file."""
        try:
            with open(self.memory_file, "rb") as f:
                self.messages = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.messages = []

def create_coaching_workflow():
    """Creates a LangGraph workflow for daily coaching.
    
    Returns:
        Graph: The configured workflow graph.
    """
    workflow = Graph()
    
    def retrieve_context(state):
        """Retrieves context from the input state.
        
        Args:
            state: The current state of the workflow.
        
        Returns:
            dict: Context information including user profile and messages.
        """
        if isinstance(state, dict) and "messages" in state:
            user_message = state["messages"][-1].content
        else:
            user_message = state[-1].content if hasattr(state, '__iter__') else state
        
        user_profile = state.get("user_profile", None)
        user_id = state.get("user_id", "default_user")
        
        return {
            "context": f"User profile: {user_profile}" if user_profile else "No profile available",
            "messages": state["messages"] if isinstance(state, dict) and "messages" in state else [state[-1]] if hasattr(state, '__iter__') else [state],
            "user_profile": user_profile,
            "user_id": user_id
        }
    
    def generate_response(state):
        """Generates a response using the daily coach agent.
        
        Args:
            state (dict): The current state including context and messages.
        
        Returns:
            dict: Response containing the AI-generated message.
        """
        try:
            context = state["context"]
            messages = state["messages"]
            user_profile = state.get("user_profile", None)
            user_id = state.get("user_id", "default_user")
            
            memory = ConversationMemory(user_id)
            memory_messages = memory.get_recent_messages(5)
            
            response = daily_coach.run(
                f"Context: {context}\n\n"
                f"User Profile: {json.dumps(user_profile, indent=2) if user_profile else 'No profile available'}\n\n"
                f"Conversation History:\n{memory_messages}\n\n"
                f"Latest Message: {messages[-1].content if messages else 'No messages'}"
            )
            
            memory.add_message("assistant", response.content)
            
            return {"messages": [AIMessage(content=response.content)]}
        except Exception as e:
            return {"messages": [AIMessage(content=f"An error occurred: {str(e)}")]}
    
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_response", generate_response)
    
    workflow.add_edge("retrieve_context", "generate_response")
    
    workflow.set_entry_point("retrieve_context")
    workflow.set_finish_point("generate_response")
    
    return workflow

coaching_workflow = create_coaching_workflow()
app = coaching_workflow.compile()

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None
if 'current_mode' not in st.session_state:
    st.session_state.current_mode = 'plan_creation'
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'progress_report' not in st.session_state:
    st.session_state.progress_report = None

# Custom CSS Styles
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #FF6347;
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: #4CAF50;
        }
        .sidebar {
            background-color: #F5F5F5;
            padding: 20px;
            border-radius: 10px;
        }
        .content {
            padding: 20px;
            background-color: #E0F7FA;
            border-radius: 10px;
            margin-top: 20px;
        }
        .btn {
            display: inline-block;
            background-color: #FF6347;
            color: white;
            padding: 10px 20px;
            text-align: center;
            border-radius: 5px;
            font-weight: bold;
            text-decoration: none;
            margin-top: 10px;
        }
        .goal-card {
            padding: 20px;
            margin: 10px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-color: #E8F5E9;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #4CAF50;
            color: #000000;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            padding: 10px;
            background-color: #E3F2FD;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .user-message {
            background-color: #4A4A4A;
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-message {
            background-color: #D3D3D3;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .progress-report {
            padding: 15px;
            background-color: #2E4053;
            color: white;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #1ABC9C;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .progress-report h3 {
            color: #1ABC9C;
            margin-top: 0;
        }
        .progress-report strong {
            color: #1ABC9C;
        }
    </style>
""", unsafe_allow_html=True)

# Main Application Interface
st.markdown('<h1 class="title">🏋️‍♂️ AI-Powered Health & Fitness Personal Trainer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Personalized fitness and nutrition plans with daily coaching!</p>', unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ Health & Fitness Inputs")
    st.subheader("Personalize Your Fitness Plan")
    
    name = st.text_input("What's your name?", "ISHOWSPEED", key="name_input")
    age = st.number_input("Age (in years)", min_value=10, max_value=100, value=25, key="age_input")
    weight = st.number_input("Weight (in kg)", min_value=30, max_value=200, value=70, key="weight_input")
    height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170, key="height_input")
    activity_level = st.selectbox("Activity Level", ["Sedentary 💻 🪑", "Lightly Active 🐌🚶‍♀️", "Occasionally Active 🚶‍♂️🧘", "Moderately Active 🚶‍♂️🏋️"], key="activity_input")
    dietary_preference = st.selectbox("Dietary Preference", ["Keto", "Vegetarian", "Low Carb", "Balanced", "Non Vegetarian"], key="diet_input")
    fitness_goal = st.selectbox("Fitness Goal", ["Weight Loss", "Muscle Gain", "Endurance", "Flexibility"], key="fitness_input")
    
    mode = st.radio("Select Mode", ["Create New Plan", "Daily Coaching"], key="mode_radio")

# Main content area
if mode == "Create New Plan":
    st.session_state.current_mode = "plan_creation"
    
    if st.sidebar.button("Generate Health Plan", key="generate_btn"):
        if not name or not age or not weight or not height:
            st.sidebar.warning("Please fill in all required fields.")
        else:
            # Create or update user profile
            user_profile = UserProfile(
                name, age, weight, height, 
                activity_level, dietary_preference, fitness_goal
            )
            st.session_state.user_profile = user_profile
            save_user_profile(user_profile)
            
            with st.spinner("💥 Generating your personalized health & fitness plan..."):
                # Generate meal plan
                meal_plan_prompt = (
                    f"Create a personalized meal plan for a {age}-year-old person, weighing {weight}kg, "
                    f"{height}cm tall, with an activity level of '{activity_level}', following a "
                    f"'{dietary_preference}' diet, aiming to achieve '{fitness_goal}'."
                )
                meal_plan = dietary_planner.run(meal_plan_prompt)
                user_profile.add_meal_plan(meal_plan.content)
                
                # Generate fitness plan
                fitness_plan_prompt = (
                    f"Generate a workout plan for a {age}-year-old person, weighing {weight}kg, "
                    f"{height}cm tall, with an activity level of '{activity_level}', "
                    f"aiming to achieve '{fitness_goal}'. Include warm-ups, exercises, and cool-downs."
                )
                fitness_plan = fitness_trainer.run(fitness_plan_prompt)
                user_profile.add_workout_plan(fitness_plan.content)
                
                # Generate full health plan
                full_health_plan = team_lead.run(
                    f"Greet the customer, {name}\n\n"
                    f"User Information: {age} years old, {weight}kg, {height}cm, activity level: {activity_level}.\n\n"
                    f"Fitness Goal: {fitness_goal}\n\n"
                    f"Meal Plan:\n{meal_plan.content}\n\n"
                    f"Workout Plan:\n{fitness_plan.content}\n\n"
                    f"Provide a holistic health strategy integrating both plans."
                )
                
                # Display the generated health plan
                st.subheader("Your Personalized Health & Fitness Plan")
                st.markdown(full_health_plan.content)
                
                st.info("This is your customized health and fitness strategy, including meal and workout plans.")
                
                # Save the profile
                save_user_profile(user_profile)
                
                # Motivational Message
                st.markdown("""
                    <div class="goal-card">
                        <h4>🏆 Stay Focused, Stay Fit!</h4>
                        <p>Consistency is key! Keep pushing yourself, and you will see results. Your fitness journey starts now!</p>
                    </div>
                """, unsafe_allow_html=True)

elif mode == "Daily Coaching":
    st.session_state.current_mode = "daily_coaching"
    
    # Load user profile
    user_profile = load_user_profile(name)
    if user_profile:
        st.session_state.user_profile = user_profile
        st.sidebar.success(f"Welcome back, {name}!")
        
        # Initialize memory
        if st.session_state.memory is None:
            st.session_state.memory = ConversationMemory(name.lower().replace(" ", "_"))
    else:
        st.sidebar.warning("No profile found. Please create a plan first.")
        st.session_state.current_mode = "plan_creation"
        st.stop()
    
    # Daily Coaching Chat Interface
    st.subheader("Daily Coaching Chat")
    
    # Chat container for messages
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.memory.get_recent_messages(10):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">You: {msg["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">Coach: {msg["content"]}</div>', 
                          unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    def clear_input():
        """Clears the chat input field."""
        st.session_state.user_input = ""
    
    user_input = st.text_input(
        "Message your coach:",
        key="chat_input",
        value=st.session_state.user_input
    )
    
    col1, col2 = st.columns(2)
    with col1:
        send_button = st.button("Send", key="send_btn")
    with col2:
        st.button("Clear", key="clear_btn", on_click=clear_input)
    
    if send_button and user_input:
        # Add user message to memory
        st.session_state.memory.add_message("user", user_input)
        
        # Generate response
        try:
            response = app.invoke({
                "user_id": name.lower().replace(" ", "_"),
                "messages": [HumanMessage(content=user_input)],
                "user_profile": user_profile.to_dict() if user_profile else None
            })
            
            # Extract response content
            response_content = response["messages"][0].content
            st.session_state.memory.add_message("assistant", response_content)
            
            # Clear input
            clear_input()
            
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
    
    # Progress Tracking
    if st.session_state.user_profile:
        with st.sidebar:
            st.markdown("---")
            st.subheader("Progress Tracking")
            new_weight = st.number_input(
                "Current Weight (kg)",
                value=st.session_state.user_profile.weight,
                key="current_weight"
            )
            progress_notes = st.text_area("Progress Notes", key="progress_notes")
            
            if st.button("Check In", key="checkin_btn"):
                st.session_state.user_profile.add_check_in(new_weight, progress_notes)
                save_user_profile(st.session_state.user_profile)
                st.success("Progress saved!")
                
                # Generate progress report
                profile_data = st.session_state.user_profile.to_dict()
                progress_report = progress_tracker.run(
                    f"User Profile:\nName: {profile_data['name']}\n"
                    f"Age: {profile_data['age']}\nWeight: {profile_data['weight']}kg\n"
                    f"Height: {profile_data['height']}cm\n"
                    f"Activity Level: {profile_data['activity_level']}\n"
                    f"Dietary Preference: {profile_data['dietary_preference']}\n"
                    f"Fitness Goal: {profile_data['fitness_goal']}\n"
                    f"Latest Check-ins:\n{profile_data['check_ins'][-3:] if profile_data['check_ins'] else 'No check-ins yet'}\n\n"
                    f"Provide a progress analysis and recommendations."
                )
                st.session_state.progress_report = progress_report.content
                
                # Add progress report to conversation
                st.session_state.memory.add_message(
                    "assistant",
                    f"Progress Report:\n{progress_report.content}"
                )
    
    # Display Progress Report
    if st.session_state.progress_report:
        st.subheader("📊 Progress Report")
        st.markdown(
            f'<div class="progress-report"><h3>Progress Report</h3>{st.session_state.progress_report}</div>',
            unsafe_allow_html=True
        )
