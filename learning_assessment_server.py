from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal
import httpx
import json
from datetime import datetime
import logging
import sqlite3
from ollama import chat

# Register the adapter
sqlite3.register_adapter(datetime, lambda x: x.isoformat())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize database and verify schema on startup"""
    logger.info("Running startup initialization...")
    try:
        init_db()
        if verify_db_schema():
            logger.info("Database schema verified successfully")
        else:
            logger.error("Database schema verification failed")
            raise SystemExit("Database initialization failed")
    except Exception as e:
        logger.error(f"Startup initialization failed: {str(e)}")
        raise SystemExit(f"Startup initialization failed: {str(e)}")

# Pydantic models for structured outputs
class CourseRecommendation(BaseModel):
    id: Optional[int] = None
    title: str
    description: str
    confidence_score: float

class CourseRecommendations(BaseModel):
    recommendations: List[CourseRecommendation]
    user_id: int

class ActivityComponent(BaseModel):
    id: str
    title: str
    description: str
    learning_objectives: Optional[List[str]] = []
    duration: Optional[str] = "1 hour"
    difficulty_level: Optional[Literal["beginner", "intermediate", "advanced"]] = "intermediate"
    prerequisites: Optional[List[str]] = []
    assessment_criteria: Optional[List[str]] = []
    type: Optional[str] = None

class SubModule(BaseModel):
    id: str
    title: str
    description: str
    activities: List[ActivityComponent]
    estimated_duration: Optional[str] = "3 hours"
    learning_outcomes: Optional[List[str]] = []

class ModuleActivity(BaseModel):
    id: str
    title: str
    description: str
    sub_modules: List[SubModule]
    duration: Optional[str] = "1 week"
    objectives: Optional[List[str]] = []
    difficulty_level: Optional[Literal["beginner", "intermediate", "advanced"]] = "intermediate"
    prerequisites: Optional[List[str]] = []

class LearningPath(BaseModel):
    modules: List[ModuleActivity]
    estimated_completion_time: Optional[str] = "4 weeks"
    prerequisites: Optional[List[str]] = []
    user_pace: Optional[str] = "normal"
    quiz_adaptations: Optional[List[str]] = []

class ContentItem(BaseModel):
    id: str
    type: Literal["text", "video", "interactive", "exercise"]
    title: str
    content: str
    duration: Optional[str] = "30 minutes"
    difficulty: Optional[Literal["basic", "intermediate", "advanced"]] = "intermediate"
    learning_objectives: Optional[List[str]] = []
    quiz_related_focus: Optional[List[str]] = None
    parent_component_id: str

class ContentModule(BaseModel):
    id: str
    title: str
    content: List[ContentItem]
    learning_objectives: Optional[List[str]] = []
    estimated_completion: Optional[str] = "1 week"
    parent_module_id: str

# Pydantic models for request validation
class SurveyResponse(BaseModel):
    careerField: str
    learningMotivation: str
    preferredLearningFormat: str
    professionalStatus: str
    skillDevelopmentGoal: str
    timeAvailability: str
    learningChallenges: str
    onlineLearningExperience: str
    learningExperience: str
    techComfortLevel: str

class QuizAnswer(BaseModel):
    question: str
    selectedAnswer: str
    correct: bool
    questionNumber: int
    topic: str

class UserQuizData(BaseModel):
    userId: int
    answers: List[QuizAnswer]



def init_db():
    """Initialize database with updated schema"""
    logger.info("Initializing database...")
    try:
        conn = sqlite3.connect('learning_data.db')
        c = conn.cursor()
        
        # Create tables with updated schema
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                user_id INTEGER PRIMARY KEY,
                survey_data TEXT,
                quiz_data TEXT,
                quiz_performance_summary TEXT,
                last_updated TIMESTAMP
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS course_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                course_title TEXT,
                course_description TEXT,
                confidence_score FLOAT,
                timestamp TIMESTAMP,
                quiz_influenced_modifications TEXT,
                FOREIGN KEY (user_id) REFERENCES user_data (user_id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS learning_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                course_id INTEGER,
                path_content TEXT,
                quiz_adaptations TEXT,
                user_pace TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_data (user_id),
                FOREIGN KEY (course_id) REFERENCES course_recommendations (id)
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS course_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                course_id INTEGER,
                content TEXT,
                quiz_based_modifications TEXT,
                pace_adjustments TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_data (user_id),
                FOREIGN KEY (course_id) REFERENCES course_recommendations (id)
            )
        ''')
        
        conn.commit()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def verify_db_schema():
    """Verify database schema and report status"""
    try:
        conn = sqlite3.connect('learning_data.db')
        c = conn.cursor()
        
        # Check for all required tables and columns
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [table[0] for table in c.fetchall()]
        
        expected_tables = ['user_data', 'course_recommendations', 'learning_paths', 'course_content']
        missing_tables = [table for table in expected_tables if table not in tables]
        
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
            
        # Verify user_data columns
        c.execute("PRAGMA table_info(user_data)")
        columns = [column[1] for column in c.fetchall()]
        expected_columns = ['user_id', 'survey_data', 'quiz_data', 'quiz_performance_summary', 'last_updated']
        missing_columns = [col for col in expected_columns if col not in columns]
        
        if missing_columns:
            logger.error(f"Missing columns in user_data: {missing_columns}")
            return False
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error verifying database schema: {str(e)}")
        return False

def categorize_question(question: str) -> str:
    """Categorize questions into topics based on their content"""
    question_lower = question.lower()
    
    categories = {
        'logical_reasoning': ['if all', 'then', 'reasoning', 'abstract'],
        'mathematical': ['number', 'produce', 'pattern', 'how many'],
        'learning_style': ['learn best', 'learning preference', 'information processing'],
        'problem_solving': ['problem-solving', 'approach', 'complex problem'],
        'study_habits': ['time management', 'approach', 'project'],
        'motivation': ['motivates', 'motivation']
    }

    for category, keywords in categories.items():
        if any(keyword in question_lower for keyword in keywords):
            return category
            
    return 'general'

def analyze_quiz_performance(quiz_data: dict) -> Dict:
    """Analyze quiz performance to identify strengths and weaknesses"""
    try:
        # Make sure we're working with the answers list
        answers = quiz_data.get('answers', [])
        if not answers:
            return {
                'topic_scores': {},
                'overall_score': 0,
                'weak_areas': []
            }

        # Initialize topic performance tracking
        topic_performance = {}
        
        # Analyze each answer and categorize by derived topic
        for answer in answers:
            question = answer.get('question', '')
            topic = categorize_question(question)
            
            if topic not in topic_performance:
                topic_performance[topic] = {'correct': 0, 'total': 0}
            
            topic_performance[topic]['total'] += 1
            if answer.get('correct', False):
                topic_performance[topic]['correct'] += 1
        
        # Calculate topic scores and identify weak areas
        topic_scores = {}
        weak_areas = []
        
        for topic, data in topic_performance.items():
            if data['total'] > 0:
                score = (data['correct'] / data['total']) * 100
                topic_scores[topic] = round(score, 2)  # Round to 2 decimal places
                if score < 60:
                    weak_areas.append(topic)

        # Calculate overall score
        total_correct = sum(data['correct'] for data in topic_performance.values())
        total_questions = sum(data['total'] for data in topic_performance.values())
        overall_score = round((total_correct / total_questions * 100), 2) if total_questions > 0 else 0

        # Add analysis summary
        performance_summary = {
            'topic_scores': topic_scores,
            'overall_score': overall_score,
            'weak_areas': weak_areas,
            'strengths': [topic for topic, score in topic_scores.items() if score >= 80],
            'total_questions_answered': total_questions,
            'question_distribution': {
                topic: data['total'] for topic, data in topic_performance.items()
            }
        }

        return performance_summary
        
    except Exception as e:
        logger.error(f"Error in analyze_quiz_performance: {str(e)}")
        return {
            'topic_scores': {},
            'overall_score': 0,
            'weak_areas': [],
            'strengths': [],
            'total_questions_answered': 0,
            'question_distribution': {}
        }

async def fetch_user_data(user_id: int) -> tuple:
    """Fetch and analyze user data from Spring Boot application"""
    try:
        async with httpx.AsyncClient() as client:
            survey_response = await client.get(f"http://localhost:8084/api/data/survey-responses")
            survey_data = [sr for sr in survey_response.json() if sr["userId"] == user_id][0]
            
            quiz_response = await client.get(f"http://localhost:8084/api/data/quiz-answers")
            quiz_data = [qd for qd in quiz_response.json() if qd["userId"] == user_id][0]
            
            # Analyze quiz performance
            quiz_analysis = analyze_quiz_performance(quiz_data)
            
            conn = sqlite3.connect('learning_data.db')
            c = conn.cursor()
            current_time = datetime.now()
            c.execute('''
                INSERT OR REPLACE INTO user_data 
                (user_id, survey_data, quiz_data, quiz_performance_summary, last_updated)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, json.dumps(survey_data), json.dumps(quiz_data), 
                  json.dumps(quiz_analysis), current_time))
            conn.commit()
            conn.close()
            
            return survey_data, quiz_data, quiz_analysis
    except Exception as e:
        logger.error(f"Error fetching user data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_course_recommendations(survey_data: dict, quiz_data: dict, quiz_analysis: dict, user_id: int) -> Dict:
    """Generate personalized course recommendations using Ollama with structured output"""
    try:
        response = chat(
            messages=[{
                'role': 'user',
                'content': f"""Generate course recommendations based on:
                Profile:
                - Career Field: {survey_data['careerField']}
                - Learning Motivation: {survey_data['learningMotivation']}
                - Professional Status: {survey_data['professionalStatus']}
                - Skill Goal: {survey_data['skillDevelopmentGoal']}
                
                Quiz Performance:
                - Overall Score: {quiz_analysis['overall_score']}%
                - Weak Areas: {', '.join(quiz_analysis['weak_areas'])}
                - Topic Scores: {json.dumps(quiz_analysis['topic_scores'], indent=2)}
                
                Learning Style: {survey_data['preferredLearningFormat']}"""
            }],
            model='llama3.2',
            format=CourseRecommendations.model_json_schema()
        )
        
        recommendations = CourseRecommendations.model_validate_json(response.message.content)
        return recommendations.model_dump()
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

def generate_component_id(prefix: str, index: int, parent_id: str = "") -> str:
    """Generate unique IDs for learning path components"""
    if parent_id:
        return f"{parent_id}-{prefix}{index}"
    return f"{prefix}{index}"

def generate_learning_path(course: dict, user_data: dict, quiz_data: dict, quiz_analysis: dict) -> Dict:
    """Generate detailed learning path with hierarchical structure and unique IDs"""
    try:
        response = chat(
            messages=[{
                'role': 'user',
                'content': f"""Create a detailed hierarchical learning path for "{course['title']}" with:
                
                Time & Experience Context:
                - Time availability: {user_data['timeAvailability']}
                - Learning challenges: {user_data['learningChallenges']}
                - Experience level: {user_data['learningExperience']}
                
                Performance Context:
                - Overall Score: {quiz_analysis['overall_score']}%
                - Weak Areas: {', '.join(quiz_analysis['weak_areas'])}
                - Topic Scores: {json.dumps(quiz_analysis['topic_scores'], indent=2)}
                
                Requirements:
                1. Create a detailed structure with modules, sub-modules, and activities
                2. Each component should have clear learning objectives
                3. Include detailed descriptions for each component
                4. Adapt difficulty based on quiz performance
                5. Consider time constraints in duration estimates
                """
            }],
            model='llama3.2',
            format=LearningPath.model_json_schema()
        )
        
        # Process the response and add IDs
        path_data = json.loads(response.message.content)
        processed_modules = []
        
        for module_idx, module in enumerate(path_data['modules']):
            module_id = generate_component_id('M', module_idx + 1)
            processed_sub_modules = []
            
            for sub_idx, sub_module in enumerate(module['sub_modules']):
                sub_module_id = generate_component_id('S', sub_idx + 1, module_id)
                processed_activities = []
                
                for act_idx, activity in enumerate(sub_module['activities']):
                    activity_id = generate_component_id('A', act_idx + 1, sub_module_id)
                    processed_activities.append({
                        **activity,
                        'id': activity_id
                    })
                
                processed_sub_modules.append({
                    **sub_module,
                    'id': sub_module_id,
                    'activities': processed_activities
                })
            
            processed_modules.append({
                **module,
                'id': module_id,
                'sub_modules': processed_sub_modules
            })
        
        path_data['modules'] = processed_modules
        return path_data
        
    except Exception as e:
        logger.error(f"Error generating learning path: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate learning path")

def generate_course_content(
    component_id: str,
    course: dict,
    learning_path: dict,
    user_data: dict,
    quiz_analysis: dict
) -> Dict:
    """Generate text-based content for a specific component of the learning path"""
    try:
        # Find the component in the learning path
        component = find_component_by_id(learning_path, component_id)
        if not component:
            raise HTTPException(status_code=404, detail="Component not found")
            
        # Safely get learning objectives with fallback to empty list
        learning_objectives = component.get('learning_objectives', [])
        
        # Safely get difficulty level with fallback to 'intermediate'
        difficulty_level = component.get('difficulty_level', 'intermediate')
        
        # Find any weak areas that match the component title
        component_title = component.get('title', '').lower()
        related_weak_areas = [
            wa for wa in quiz_analysis.get('weak_areas', [])
            if wa in component_title
        ]
            
        response = chat(
            messages=[{
                'role': 'user',
                'content': f"""Generate detailed text-based educational content for component "{component.get('title', 'Unnamed Component')}" considering:
                
                Component Context:
                - Type: {get_component_type(component_id)}
                - Learning Objectives: {json.dumps(learning_objectives)}
                - Difficulty Level: {difficulty_level}
                
                User Context:
                - Learning Experience: {user_data.get('learningExperience', 'intermediate')}
                - Time Availability: {user_data.get('timeAvailability', 'not specified')}
                
                Performance Context:
                - Related Weak Areas: {json.dumps(related_weak_areas)}
                
                Requirements:
                1. Create only text-based content divided into clear sections
                2. Each section should:
                   - Have a clear title
                   - Include detailed explanations
                   - Provide examples where appropriate
                   - End with key takeaways
                3. Content should align with learning objectives
                4. Language should match user's experience level
                5. Include practice questions or reflection points
                """
            }],
            model='llama3.2',
            format=ContentModule.model_json_schema()
        )
        
        content = json.loads(response.message.content)
        processed_content = []
        
        # Ensure all content items are text type
        for idx, item in enumerate(content['content']):
            content_id = generate_component_id('C', idx + 1, component_id)
            processed_content.append({
                'id': content_id,
                'type': 'text',
                'title': item['title'],
                'content': item['content'],
                'duration': '15 minutes',  # Default duration for text content
                'difficulty': difficulty_level,
                'learning_objectives': learning_objectives,
                'parent_component_id': component_id
            })
            
        content['content'] = processed_content
        content['id'] = f"CNT-{component_id}"
        content['parent_module_id'] = component_id
        content['learning_objectives'] = learning_objectives
        content['estimated_completion'] = f"{len(processed_content) * 15} minutes"
        
        return content
        
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        logger.error(f"Component ID: {component_id}")
        logger.error(f"Component data: {component if 'component' in locals() else 'Not found'}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate content: {str(e)}"
        )

def find_component_by_id(learning_path: dict, component_id: str) -> Optional[dict]:
    """Find a component in the learning path by its ID"""
    for module in learning_path['modules']:
        if module['id'] == component_id:
            return module
            
        for sub_module in module['sub_modules']:
            if sub_module['id'] == component_id:
                return sub_module
                
            for activity in sub_module['activities']:
                if activity['id'] == component_id:
                    return activity
                    
    return None

def get_component_type(component_id: str) -> str:
    """Determine component type from ID"""
    if component_id.startswith('M'):
        return "module"
    elif component_id.startswith('S'):
        return "sub_module"
    elif component_id.startswith('A'):
        return "activity"
    return "unknown"

@app.get("/suggest_courses/{user_id}")
async def get_suggested_courses(user_id: int):
    try:
        # Add detailed logging
        logger.info(f"Starting course suggestion for user_id: {user_id}")
        
        # Step 1: Fetch user data with error handling
        try:
            survey_data, quiz_data, quiz_analysis = await fetch_user_data(user_id)
            logger.info("Successfully fetched user data")
            logger.debug(f"Quiz Analysis: {quiz_analysis}")
        except Exception as e:
            logger.error(f"Error in fetch_user_data: {str(e)}")
            logger.error(f"Full error details: {e.__class__.__name__}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching user data: {str(e)}"
            )

        # Step 2: Generate recommendations with error handling
        try:
            recommendations = generate_course_recommendations(
                survey_data, quiz_data, quiz_analysis, user_id
            )
            logger.info("Successfully generated recommendations")
            logger.debug(f"Recommendations: {recommendations}")
        except Exception as e:
            logger.error(f"Error in generate_course_recommendations: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating recommendations: {str(e)}"
            )

        # Step 3: Store recommendations with error handling
        try:
            conn = sqlite3.connect('learning_data.db')
            c = conn.cursor()
            current_time = datetime.now()
            
            stored_recommendations = []
            for course in recommendations['recommendations']:
                c.execute('''
                    INSERT INTO course_recommendations 
                    (user_id, course_title, course_description, confidence_score, 
                     quiz_influenced_modifications, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                    RETURNING id
                ''', (
                    user_id, course['title'], course['description'],
                    course['confidence_score'],
                    json.dumps(quiz_analysis.get('weak_areas', [])),
                    current_time
                ))
                
                course_id = c.fetchone()[0]
                course['id'] = course_id
                stored_recommendations.append(course)
            
            conn.commit()
            conn.close()
            logger.info("Successfully stored recommendations")
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {str(e)}")
            if 'conn' in locals():
                conn.close()
            raise HTTPException(
                status_code=500,
                detail=f"Error storing recommendations: {str(e)}"
            )

        # Prepare and return response
        response_data = {
            "recommendations": stored_recommendations,
            "user_id": user_id,
            "quiz_performance": quiz_analysis
        }
        logger.info("Successfully completed course suggestion process")
        return response_data
        
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error in get_suggested_courses: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
@app.get("/learning_path/{user_id}/{course_id}")
async def get_learning_path(user_id: int, course_id: int):
    try:
        conn = sqlite3.connect('learning_data.db')
        c = conn.cursor()
        
        # Fetch necessary data
        c.execute('''
            SELECT survey_data, quiz_data, quiz_performance_summary 
            FROM user_data WHERE user_id = ?
        ''', (user_id,))
        user_row = c.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        
        c.execute(
            'SELECT * FROM course_recommendations WHERE id = ? AND user_id = ?', 
            (course_id, user_id)
        )
        course_row = c.fetchone()
        if not course_row:
            raise HTTPException(status_code=404, detail="Course not found")
        
        user_data = json.loads(user_row[0])
        quiz_data = json.loads(user_row[1])
        quiz_analysis = json.loads(user_row[2])
        
        course = {
            "id": course_row[0],
            "title": course_row[2],
            "description": course_row[3],
            "confidence_score": course_row[4]
        }
        
        learning_path = generate_learning_path(
            course, user_data, quiz_data, quiz_analysis
        )
        
        # Store learning path with quiz adaptations
        current_time = datetime.now()
        c.execute('''
            INSERT INTO learning_paths 
            (user_id, course_id, path_content, quiz_adaptations, user_pace, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            course_id, 
            json.dumps(learning_path),
            json.dumps(quiz_analysis['weak_areas']),
            learning_path.get('user_pace', 'normal'),  # Add default value 'normal'
            current_time
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "learning_path": learning_path,
            "course_id": course_id,
            "user_id": user_id,
            "quiz_performance": quiz_analysis
        }
        
    except Exception as e:
        logger.error(f"Error in get_learning_path: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/course_content/{user_id}/{course_id}/{component_id}")
async def get_component_content(user_id: int, course_id: int, component_id: str):
    try:
        conn = sqlite3.connect('learning_data.db')
        c = conn.cursor()
        
        # Fetch necessary data
        c.execute('''
            SELECT survey_data, quiz_data, quiz_performance_summary 
            FROM user_data WHERE user_id = ?
        ''', (user_id,))
        user_row = c.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        
        c.execute('''
            SELECT * FROM course_recommendations 
            WHERE id = ? AND user_id = ?
        ''', (course_id, user_id))
        course_row = c.fetchone()
        if not course_row:
            raise HTTPException(status_code=404, detail="Course not found")
        
        c.execute('''
            SELECT path_content FROM learning_paths 
            WHERE course_id = ? AND user_id = ?
        ''', (course_id, user_id))
        path_row = c.fetchone()
        if not path_row:
            raise HTTPException(status_code=404, detail="Learning path not found")
        
        # Parse data
        user_data = json.loads(user_row[0])
        quiz_analysis = json.loads(user_row[2])
        learning_path = json.loads(path_row[0])
        
        course = {
            "id": course_row[0],
            "title": course_row[2],
            "description": course_row[3]
        }
        
        # Generate content for specific component
        content = generate_course_content(
            component_id,
            course,
            learning_path,
            user_data,
            quiz_analysis
        )
        
        # Store the generated content
        current_time = datetime.now()
        c.execute('''
            INSERT INTO course_content 
            (user_id, course_id, content, quiz_based_modifications, 
             pace_adjustments, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            course_id,
            json.dumps(content),
            json.dumps(quiz_analysis['weak_areas']),
            learning_path.get('user_pace', 'normal'),
            current_time
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "component_content": content,
            "component_id": component_id,
            "course_id": course_id,
            "user_id": user_id,
            "performance_context": {
                "quiz_performance": quiz_analysis,
                "component_type": get_component_type(component_id)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in get_component_content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if check_database_connection() else "disconnected"
    }

def check_database_connection() -> bool:
    try:
        conn = sqlite3.connect('learning_data.db')
        conn.cursor()
        conn.close()
        return True
    except Exception:
        return False

# Error handling middleware
@app.middleware("http")
async def add_error_handling(request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred",
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('server.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Initializing database...")
    init_db()
    
    if verify_db_schema():
        logger.info("Database schema verified successfully")
    else:
        logger.error("Database schema verification failed")
        raise SystemExit("Database initialization failed")
    
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="debug",
        reload=True
    )
    