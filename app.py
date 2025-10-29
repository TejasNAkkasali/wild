# merged_smart_waste_system.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from PIL import Image
import hashlib
import random
import os, datetime as dt

# Optional AI libs (CLIP + torch + cv2). If missing, fallback to heuristics.
try:
    import torch
    import clip
    import torchvision.transforms as T
    import cv2
    CLIP_AVAILABLE = True
except Exception as e:
    CLIP_AVAILABLE = False
    # Attempt to import cv2 separately if available
    try:
        import cv2
    except:
        cv2 = None

# --- Data directory for uploaded images / persistence ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- User Authentication Functions ---
USERS = {
    # Role: User/Citizen
    "citizen": hashlib.sha256("i_recycle_2025".encode()).hexdigest(),
    # Role: Area Leader
    "leader": hashlib.sha256("area_manage_1!".encode()).hexdigest(),
    # Role: Municipality
    "municipality": hashlib.sha256("city_planner_#".encode()).hexdigest(),
}

USER_ROLES = {
    "citizen": "User/Citizen",
    "leader": "Area Leader",
    "municipality": "Municipality",
}

def check_password(username, password):
    """Verifies the username and password against the mock database."""
    if username in USERS:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return hashed_password == USERS[username]
    return False

# -------------------------------
# CLIP-based AI helpers (from first code)
# -------------------------------
if CLIP_AVAILABLE:
    @st.cache_resource
    def load_clip_model():
        """Load CLIP model (cached)."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device
else:
    # placeholder function to avoid NameError
    def load_clip_model():
        return None, None, None

def rate_garbage_ai(img_path):
    """
    Rate image cleanliness using CLIP similarity to 'garbage' and 'clean' prompts.
    Returns (ai_rating (float 1-5), model_confidence float 0-1, note string)
    If CLIP unavailable, returns heuristic fallback rating.
    """
    if CLIP_AVAILABLE:
        model, preprocess, device = load_clip_model()
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            text_prompts = [
                "a clean street with no garbage",
                "a slightly littered area",
                "an area with moderate garbage",
                "an area with a lot of garbage",
                "a very dirty place full of trash"
            ]
            text_tokens = clip.tokenize(text_prompts).to(device)

            with torch.no_grad():
                logits_per_image, _ = model(image, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            ai_rating = sum((i + 1) * p for i, p in enumerate(probs))
            # confidence estimated as the max softmax prob
            model_confidence = float(probs.max())
            return round(ai_rating, 1), model_confidence, "CLIP"
        except Exception as e:
            # fallback to heuristic if something goes wrong running CLIP
            return rate_garbage_heuristic(img_path, note=f"CLIP error: {e}")
    else:
        return rate_garbage_heuristic(img_path, note="CLIP not available")

def rate_garbage_heuristic(img_path, note="heuristic"):
    """Simple fallback: use OpenCV edge-density heuristic to map to 1-5 scale."""
    if cv2 is None:
        # absolute fallback: random-ish deterministic mapping based on file size
        size = os.path.getsize(img_path)
        score = 1 + (size % 5)
        return float(score), 0.5, note
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2.imread returned None")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 150)
        clutter = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        # map clutter (0 -> 0.2) -> rating 1..5
        normalized = min(max((clutter / 0.08), 0.0), 1.0)
        ai_rating = 1.0 + normalized * 4.0
        model_confidence = 0.6 + (0.4 * (1 - abs(0.04 - clutter) / 0.04)) if clutter <= 0.08 else 0.6
        return round(ai_rating,1), float(model_confidence), note
    except Exception as e:
        # ultimate fallback
        return 3.0, 0.4, f"heuristic_failed:{e}"

def analyze_image_garbage(img_path):
    """Detect clutter using simple edge-based heuristic (same as earlier).
       Returns a text verdict and a numeric contamination_score (0-1)."""
    if cv2 is None:
        return "Analysis unavailable (cv2 missing)", 0.5
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("cv2.imread returned None")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 150)
        clutter = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        verdict = "High clutter" if clutter > 0.05 else "Low clutter"
        contamination_score = float(min(max(clutter * 2.0, 0.0), 1.0))  # scale to 0-1
        return verdict, contamination_score
    except Exception as e:
        return f"Analysis failed: {e}", 0.5

# -------------------------------
# Keep older mocks (other modules) — unchanged
# -------------------------------
def mock_predictive_analytics(area, forecast_days):
    time.sleep(1)
    dates = pd.date_range(end=pd.Timestamp.now(), periods=forecast_days, freq='D')
    forecast = pd.DataFrame({
        'Date': dates,
        'Predicted_Weight_kg': np.random.randint(1500, 3500, forecast_days) * (1 + np.random.normal(0, 0.1, forecast_days)),
    })
    optimized_route = [
        (40.7128, -74.0060, "Start Depot"),
        (40.7580, -73.9855, "Area A Stop 1"),
        (40.7484, -73.9857, "Area B Stop 2"),
        (40.7000, -74.0200, "End Depot")
    ]
    return forecast, optimized_route

def mock_nlp_analysis(text):
    time.sleep(0.5)
    if "missed" in text.lower() or "not collected" in text.lower() or "terrible" in text.lower():
        sentiment = "Negative 😠"
        category = "Pickup Complaint"
    elif "great" in text.lower() or "love" in text.lower() or "excellent" in text.lower():
        sentiment = "Positive 😊"
        category = "General Feedback"
    elif "bin is full" in text.lower() or "overflowing" in text.lower():
        sentiment = "Negative 😠"
        category = "Overfill Alert"
    else:
        sentiment = "Neutral 😐"
        category = "Query/Other"
    return sentiment, category

def mock_recommendation_engine(user_id):
    time.sleep(0.3)
    recommendations = {
        101: "DIY Plastic Bottle Vertical Garden",
        102: "Marketplace: Refurbished Laptop Stand (from E-Waste)",
        103: "Project: Paper Mache Art from Cardboard"
    }
    seed = user_id % 3
    keys = list(recommendations.keys())
    return [recommendations[keys[(i + seed) % len(keys)]] for i in range(3)]

def mock_chatbot_response(user_input):
    time.sleep(0.8)
    user_input_lower = user_input.lower()
    if "recycle" in user_input_lower or "electronic" in user_input_lower or "ewaste" in user_input_lower:
        responses = [
            "You can take electronic waste to any designated drop-off center. Please check our map for the nearest location.",
            "E-waste recycling points are listed on the 'E-Waste Diagnosis' module. Make sure to remove any batteries first!",
            "For large electronics, schedule a special pickup. Small items go to the designated e-waste bins."
        ]
        return random.choice(responses)
    elif "policy" in user_input_lower or "rules" in user_input_lower:
        responses = [
            "Our current waste sorting policy requires rinsing all plastic and glass before recycling.",
            "Check the official municipal guide on our website for the latest policy updates on textiles and organics.",
            "Items smaller than 2 inches should typically be placed in the general waste unless specified otherwise."
        ]
        return random.choice(responses)
    elif "collection" in user_input_lower or "day" in user_input_lower or "when" in user_input_lower:
        responses = [
            "Collection day for your area is usually Tuesday, but please confirm the time with your address in the system.",
            "Missed collection? Please log a complaint through the 'Community Board' and we will re-route a truck.",
            "Our trucks typically operate from 7:00 AM to 4:00 PM in your zone."
        ]
        return random.choice(responses)
    elif "cost" in user_input_lower or "fee" in user_input_lower or "charge" in user_input_lower:
        responses = [
            "Standard residential recycling is covered by municipal taxes. Fees apply only for bulk waste or special pickups.",
            "There is a $50 fee for removing construction debris. Regular waste disposal is free.",
            "Tipping fees are only relevant for commercial entities. Residential users have no direct charges."
        ]
        return random.choice(responses)
    else:
        responses = [
            "I am an AI assistant for waste management. How can I help you with classification, collection, or policy questions?",
            "Could you please rephrase your query? I can assist with topics like recycling rules, collection schedules, or material disposal.",
            "Thank you for reaching out! What specific information about waste management are you looking for?"
        ]
        return random.choice(responses)

def mock_ewaste_diagnosis(image_data):
    time.sleep(1.5)
    score = np.random.randint(3, 9)
    features = {
        "screws_detected": 4,
        "adhesive_area_px": 500,
        "modularity_score": 0.8
    }
    st.session_state['ewaste_diagnosis_history'] = {
        'score': score,
        'features': features,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'diagnoser': st.session_state['username']
    }
    return score, features

def mock_rl_reward_allocation(user_state):
    time.sleep(0.1)
    if user_state["contamination_score_history"] < 0.1:
        points = 20
    elif user_state["contamination_score_history"] < 0.3:
        points = 15
    else:
        points = 5
    return points

def generate_mock_leaderboard():
    users = [f"User{i:03d}" for i in range(1, 11)]
    data = {
        'User ID': users,
        'Contamination Score': np.around(np.sort(np.random.uniform(0.05, 0.4, 10)), decimals=3),
        'Total Volume (m³ )': np.around(np.sort(np.random.uniform(5, 50, 10))[::-1], decimals=2),
        'Rewards Points': np.random.randint(100, 500, 10)
    }
    df = pd.DataFrame(data)
    df['Contamination Rank'] = df['Contamination Score'].rank(method='min').astype(int)
    df['Volume Rank'] = df['Total Volume (m³ )'].rank(method='min', ascending=False).astype(int)
    return df.sort_values(by='Contamination Rank', ascending=True)

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="AI-Powered Waste Management System (Merged CLIP)")
st.title("🌱 AI-Powered Waste Management Command Center (CLIP-integrated)")
st.markdown("---")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'operational_log' not in st.session_state:
    st.session_state['operational_log'] = []
if 'municipality_notifications' not in st.session_state:
    st.session_state['municipality_notifications'] = []
if 'ewaste_diagnosis_history' not in st.session_state:
    st.session_state['ewaste_diagnosis_history'] = None

# Login UI
if not st.session_state['authenticated']:
    st.sidebar.header("🔐 User Login")
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if check_password(username_input, password_input):
            st.session_state['authenticated'] = True
            st.session_state['username'] = username_input
            st.session_state['role'] = USER_ROLES.get(username_input, "Unknown")
            st.toast(f"Welcome, {st.session_state['role']} ({username_input})!")
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password.")
    st.sidebar.info("""
    Mock Credentials (Username/Password):
    - citizen / i_recycle_2025
    - leader / area_manage_1!
    - municipality / city_planner_#
    """)
    st.warning("Please log in to access the AI Command Center.")

# Main application (authenticated)
if st.session_state['authenticated']:

    # Logout
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.rerun()

    st.sidebar.success(f"Logged in as: {st.session_state['role']}")
    st.sidebar.markdown("---")

    # Define modules & role access (unchanged)
    ALL_MODULES = {
        "1. Computer Vision (CV)": ["User/Citizen", "Area Leader", "Municipality"],
        "2. Predictive Analytics & Route Opt.": ["Area Leader", "Municipality"],
        "3. NLP (Community Board)": ["User/Citizen", "Area Leader", "Municipality"],
        "4. Recommendation Engine": ["User/Citizen"],
        "6. Chatbot (English Only)": ["User/Citizen", "Area Leader", "Municipality"],
        "7. E-Waste Diagnosis": ["User/Citizen", "Municipality"],
        "8. Dynamic Rewards (RL)": ["User/Citizen"],
        "9. Leader Dashboard (Gamification)": ["User/Citizen", "Area Leader", "Municipality"],
    }

    user_role = st.session_state['role']
    accessible_modules = [
        name for name, roles in ALL_MODULES.items() if user_role in roles
    ]

    st.sidebar.title("System Capabilities")
    if accessible_modules:
        selected_capability = st.sidebar.radio("Select AI Module:", accessible_modules)
    else:
        st.sidebar.warning("No modules available for this role.")
        selected_capability = None

    st.sidebar.markdown("---")
    st.sidebar.info(f"Your role ({user_role}) determines the modules you can access.")

    # --- 1. Computer Vision (CV) (now using CLIP + OpenCV heuristic) ---
    if selected_capability == "1. Computer Vision (CV)":
        st.header("1. Computer Vision: Waste Analysis & Uploads")
        st.subheader("CLIP-based Severity Rating + Visual Heuristics")

        # --- Citizen submission UI ---
        if user_role == "User/Citizen":
            st.markdown("### 📸 Submit Waste Photo for Classification")
            uploaded_file = st.file_uploader("Upload an image of waste to classify:", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # save file locally
                timestamp_str = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
                filename = f"{st.session_state['username']}report{timestamp_str}.jpg"
                img_path = os.path.join(DATA_DIR, filename)
                with open(img_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                image = Image.open(img_path)
                st.image(image, caption='Image for Analysis', use_container_width=True)

                if st.button("Run CV Analysis & Log Data"):
                    with st.spinner('Running CLIP + heuristics...'):
                        ai_rating, model_confidence, model_note = rate_garbage_ai(img_path)
                        verdict, contamination_score = analyze_image_garbage(img_path)

                        # classification label from ai_rating
                        if ai_rating <= 2.0:
                            classification = "Clean/Low"
                        elif ai_rating <= 3.5:
                            classification = "Moderate"
                        else:
                            classification = "Heavy"

                        # estimate volume using heuristic (small random but deterministic-ish)
                        volume_estimate_m3 = max(0.01, round(0.02 + (contamination_score * 0.08) + random.uniform(-0.005, 0.01), 4))

                        # confidence fallback for CLIP availability
                        confidence_display = f"{model_confidence:.2f} ({model_note})"

                        # log into operational_log (shared mock DB)
                        st.session_state['operational_log'].insert(0, {
                            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Uploader": st.session_state['username'],
                            "Classification": classification,
                            "Contamination Score": f"{contamination_score:.2f}",
                            "Volume (m³ )": f"{volume_estimate_m3:.3f}",
                            "AI Rating": f"{ai_rating}/5",
                            "Confidence": confidence_display,
                            "Image Path": img_path
                        })

                        st.success("✅ Analysis Complete! Data Logged.")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Classification", classification)
                        col2.metric("Contamination Score", f"{contamination_score:.2f}")
                        col3.metric("AI Rating (1-5)", f"{ai_rating}/5")

                        # Visual guidance
                        if ai_rating <= 2:
                            st.info("🟢 Clean Area Detected")
                        elif ai_rating <= 3.5:
                            st.warning("🟡 Moderate Garbage Detected")
                        else:
                            st.error("🔴 Heavy Garbage Detected — Needs Urgent Action!")

            st.markdown("---")
            if st.session_state['operational_log']:
                st.subheader(f"Your Latest Submission: {st.session_state['username']}")
                user_logs = [log for log in st.session_state['operational_log'] if log['Uploader'] == st.session_state['username']]
                if user_logs:
                    st.dataframe(pd.DataFrame([user_logs[0]]), use_container_width=True)
                else:
                    st.info("You haven't submitted any photos yet.")

        # --- Leader & Municipality views of operational log ---
        if user_role in ["Area Leader", "Municipality"]:
            st.markdown("### 📈 Operational Log: Real-Time Monitoring")
            st.info("This table shows all citizen/user waste submissions. Use this data for oversight and planning.")

            if st.session_state['operational_log']:
                log_df = pd.DataFrame(st.session_state['operational_log'])
                st.dataframe(log_df, use_container_width=True)

                st.markdown("#### Aggregated Insights")
                colA, colB = st.columns(2)
                colA.metric("Total Submissions", len(log_df))

                valid_contamination = []
                try:
                    valid_contamination = log_df[log_df['Contamination Score'] != '-']['Contamination Score'].astype(float)
                except:
                    valid_contamination = pd.Series(dtype=float)

                colB.metric("Average Contamination", f"{valid_contamination.mean():.2f}" if not valid_contamination.empty else "N/A")

                if st.button("Clear Operational Log (Admin Only)", help="Clears the mock shared database."):
                    st.session_state['operational_log'] = []
                    st.rerun()
            else:
                st.info("The operational log is currently empty.")

    # --- 2. Predictive Analytics & Route Opt. ---
    elif selected_capability == "2. Predictive Analytics & Route Opt.":
        st.header("2. Predictive Analytics & Route Optimization")
        col1, col2 = st.columns(2)
        with col1:
            area_select = st.selectbox("Select Area for Forecasting:", ["Downtown Core", "Residential East", "Industrial Park"])
            forecast_days = st.slider("Forecast Horizon (Days):", 7, 60, 30)
            if st.button("Generate Forecast & Optimize Route"):
                forecast_df, route_coords = mock_predictive_analytics(area_select, forecast_days)
                st.session_state['forecast_data'] = forecast_df
                st.session_state['route_coords'] = route_coords
        if 'forecast_data' in st.session_state:
            st.subheader(f"Waste Generation Forecast for {area_select}")
            st.line_chart(st.session_state['forecast_data'].set_index('Date'))
            st.subheader("Optimal Collection Route (VRP Solver)")
            route_df = pd.DataFrame(st.session_state['route_coords'], columns=['lat', 'lon', 'Stop'])
            st.map(route_df)
            st.markdown("Optimized Stops:")
            st.dataframe(route_df, use_container_width=True)
            st.info("💡 Algorithm: SARIMA/Prophet for forecasting, Google OR-Tools (VRP) for route planning.")

    # --- 3. NLP (Community Board) ---
    elif selected_capability == "3. NLP (Community Board)":
        st.header("3. NLP: Community Board Analysis")
        st.subheader("Real-Time Sentiment & Auto-Categorization (DistilBERT)")

        if user_role != "Municipality":
            user_prompt = "Post your feedback/query here:"
            default_text = "The recycling bin is overflowing and hasn't been collected in three days."
            if user_role == "Area Leader":
                user_prompt = "Post an operational alert/query:"
                default_text = "Need to reroute truck A due to blockage at Sector 4."
            new_post = st.text_area(user_prompt, default_text, height=150)
            if st.button("Analyze & Submit Post"):
                st.markdown("---")
                sentiment, category = mock_nlp_analysis(new_post)
                notification_data = {
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "User": st.session_state['username'],
                    "Role": user_role,
                    "Category": category,
                    "Sentiment": sentiment,
                    "Message Snippet": new_post[:50] + "..."
                }
                st.session_state['municipality_notifications'].insert(0, notification_data)
                col1, col2 = st.columns(2)
                col1.metric("Sentiment Analysis", sentiment)
                col2.metric("Auto-Categorization", category)
                st.success("✅ Analysis complete. The post has been automatically categorized and a real-time notification sent to the Municipality.")
                st.info("💡 Model: Lightweight model like DistilBERT for low-latency real-time processing.")
            st.markdown("---")

        if user_role == "User/Citizen":
            st.subheader("Your Recent Posts")
            user_posts = [log for log in st.session_state['municipality_notifications'] if log['User'] == st.session_state['username']]
            if user_posts:
                st.dataframe(pd.DataFrame(user_posts), use_container_width=True)
            else:
                st.info("No recent posts found from you.")
        elif user_role == "Area Leader":
            st.subheader("All Recent Community Posts")
            st.info("As a Leader, you see all recent analyzed posts to monitor neighborhood issues.")
            if st.session_state['municipality_notifications']:
                st.dataframe(pd.DataFrame(st.session_state['municipality_notifications']), use_container_width=True)
            else:
                st.info("The community board is currently quiet.")
        elif user_role == "Municipality":
            st.markdown("---")
            st.header("🔔 Real-Time Community Notifications")
            st.warning(f"Total New Alerts: {len(st.session_state['municipality_notifications'])}")
            if st.session_state['municipality_notifications']:
                notifications_df = pd.DataFrame(st.session_state['municipality_notifications'])
                st.dataframe(notifications_df, use_container_width=True)
                st.markdown("#### Notification Summary")
                colA, colB = st.columns(2)
                colA.metric("Negative Sentiment Alerts", len(notifications_df[notifications_df['Sentiment'].str.contains('Negative')]))
                colB.metric("Pickup Complaints", len(notifications_df[notifications_df['Category'] == 'Pickup Complaint']))
                if st.button("Acknowledge & Clear Notifications"):
                    st.session_state['municipality_notifications'] = []
                    st.rerun()
            else:
                st.info("No new community notifications to display.")

    # --- 4. Recommendation Engine ---
    elif selected_capability == "4. Recommendation Engine":
        st.header("4. Recommendation Engine: Personalized Suggestions")
        st.subheader("Collaborative Filtering (ALS) for Upcycling")
        user_id = 1
        if st.button(f"Generate Recommendations for {st.session_state['username']}"):
            recommendations = mock_recommendation_engine(user_id)
            st.success(f"Top 3 personalized suggestions for you!")
            cols = st.columns(3)
            for i, rec in enumerate(recommendations):
                with cols[i]:
                    st.subheader(f"#{i+1}")
                    st.markdown(f"{rec}")
                    st.button(f"View Details {i+1}", key=f"rec_btn_{i}")
            st.info("💡 Algorithm: Alternating Least Squares (ALS) Matrix Factorization on a User-Item interaction matrix.")
            if user_id in [1, 2]:
                st.markdown("---")
                st.warning("⚠ Cold Start Warning: For new users, we fall back to suggesting the overall most popular item (DIY Bird Feeder).")

    # --- 6. Chatbot (English Only) ---
    elif selected_capability == "6. Chatbot (English Only)":
        st.header("6. Conversational Assistant (English Only)")
        st.subheader("RAG/LLM for User Guidance")
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [{"role": "assistant", "content": "Hello! I am your AI waste assistant. How can I help you today? Try asking about 'collection day' or 'e-waste rules'."}]
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        user_input = st.chat_input(f"Speak to the assistant...")
        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = mock_chatbot_response(user_input)
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            if len(st.session_state["chat_messages"]) > 8:
                st.sidebar.warning("⚠ Fallback: Multiple failed attempts detected. Offering human handoff...")

    # --- 7. E-Waste Diagnosis ---
    elif selected_capability == "7. E-Waste Diagnosis":
        st.header("7. E-Waste Diagnosis: Repairability Assessment")
        st.subheader("Sequential Vision Pipeline & Scoring Model")
        diagnosis_results = st.session_state.get('ewaste_diagnosis_history')

        # Uploader (Citizen)
        if user_role == "User/Citizen":
            uploaded_file_ewaste = st.file_uploader("Upload image of E-Waste (e.g., smartphone interior):", type=["jpg", "jpeg", "png"])
            if uploaded_file_ewaste is not None:
                image = Image.open(uploaded_file_ewaste)
                st.image(image, caption='Device for Diagnosis', use_container_width=True)
                if st.button("Assess Repairability"):
                    with st.spinner('Running multi-step vision pipeline...'):
                        mock_ewaste_diagnosis(uploaded_file_ewaste.getvalue())
                        st.success("✅ Diagnosis Complete! Results are logged.")
                        st.rerun()
            st.markdown("---")
            if diagnosis_results:
                st.subheader(f"Your Last Diagnosis ({diagnosis_results['timestamp']})")

        # Viewer (Citizen & Municipality)
        if diagnosis_results:
            score = diagnosis_results['score']
            features = diagnosis_results['features']
            if user_role == "Municipality":
                st.subheader(f"Latest E-Waste Diagnosis for Oversight ({diagnosis_results['timestamp']})")
                st.info(f"Diagnosis performed by: {diagnosis_results['diagnoser']}")
            st.subheader(f"Repairability Score: {score}/10")
            st.progress(score / 10.0)
            st.markdown("---")
            st.subheader("🛠 User-Friendly Repairability Breakdown")
            repair_data = {
                "Metric": ["Screws Detected", "Adhesive Area (px)", "Modularity Score"],
                "Technical Value": [features["screws_detected"], features["adhesive_area_px"], features["modularity_score"]],
                "User-Friendly Interpretation": [
                    "Low/Moderate Fastening. Secured with screws, easy to open.",
                    "Minimal Adhesive Use. Very little glue, reduces risk of component breakage.",
                    "High Modularity. Key parts are connected with simple plugs, easy to replace individually."
                ],
                "Impact on Repair": ["Positive", "Positive", "Strong Positive"]
            }
            st.table(pd.DataFrame(repair_data))
            st.markdown("### 💰 Estimated Repair Cost (Mock)")
            st.info(f"Based on the high repairability indicators (Modularity: {features['modularity_score']}), labor time is low.")
            cost_data = {
                "Item": ["Replacement Part (Mock)", "Labor (1-2 Hrs)"],
                "Estimated Cost (USD)": ["$50 - $150", "$75 - $150"],
                "Notes": ["Assumes battery/screen failure.", "Low labor due to screw-fastened, modular design."]
            }
            st.dataframe(pd.DataFrame(cost_data), hide_index=True, use_container_width=True)
            st.markdown("Total Estimated Repair Cost: $125 - $300")
            st.info("💡 Model Pipeline: Object Detection (Components) → Feature Extraction (Screws/Adhesive) → Decision Tree (Scoring).")
        else:
            st.info("No E-Waste diagnosis history is currently available.")

    # --- 8. Dynamic Rewards (RL) ---
    elif selected_capability == "8. Dynamic Rewards (RL)":
        st.header("8. Dynamic Rewards: Optimization via RL")
        st.subheader("Reinforcement Learning (A2C/DQN) Policy for Point Allocation")
        current_user_contamination = st.slider("User's Historical Contamination Score:", 0.0, 1.0, 0.2, 0.05)
        user_state = {
            "user_id": 123,
            "contamination_score_history": current_user_contamination,
            "weekly_recycling_count": 5
        }
        if st.button("Allocate Reward Points"):
            points_allocated = mock_rl_reward_allocation(user_state)
            st.success("✅ Reward Policy Executed!")
            st.metric("Points Allocated", f"{points_allocated} Points", delta="Dynamic reward based on RL Policy")
            st.markdown(f"""
                <div style='background-color: #e6f7ff; padding: 15px; border-radius: 5px;'>
                Based on the current State (Contamination: {current_user_contamination:.2f}), 
                the trained RL Policy (A2C/DQN) determined the optimal action was to award {points_allocated} points to maximize 
                long-term system utility (i.e., less contamination).
                </div>
                """, unsafe_allow_html=True)
            if points_allocated == 5:
                st.warning("⚠ Low Reward Allocation: User's history suggests high contamination. Policy assigned a low reward to discourage poor behavior. Fallback is a fixed 10 points.")

    # --- 9. Leader Dashboard (Gamification) ---
    elif selected_capability == "9. Leader Dashboard (Gamification)":
        st.header("9. Leader Dashboard: Community Recycling Performance")
        st.subheader("Ranking Users by Contamination Score and Volume")
        leaderboard_df = generate_mock_leaderboard()
        st.markdown("### 🥇 Overall Performance (Ranked by Lowest Contamination)")
        st.dataframe(
            leaderboard_df[['Contamination Rank', 'User ID', 'Contamination Score', 'Total Volume (m³ )', 'Rewards Points']],
            use_container_width=True,
            column_config={
                "Contamination Rank": st.column_config.NumberColumn(
                    "Rank",
                    format="%d",
                    help="Lower Contamination Score = Higher Rank"
                )
            }
        )
        st.info("💡 Insight: This dashboard drives user engagement and accountability by publicly recognizing top performers. It utilizes data aggregated from the CV, RL, and other systems.")
        st.markdown("---")
        st.markdown("### 📈 Top Recyclers (Ranked by Total Volume)")
        top_volume_df = leaderboard_df.sort_values(by='Volume Rank', ascending=True)
        st.table(top_volume_df[['Volume Rank', 'User ID', 'Total Volume (m³ )']].head(5))

# Footer
st.markdown("---")
st.caption("Merged: CLIP-based AI (or heuristic fallback) integrated into the multi-module Waste Management system.")
