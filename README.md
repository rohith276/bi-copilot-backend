# BI-Copilot Backend

AI-powered Business Intelligence backend built with FastAPI, SQLAlchemy, and OpenAI.

## Deployment to Render

1. **Create Repo**: Create a new private GitHub repository and push the contents of this `backend/` folder to it.
2. **Deploy**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Create a **New Web Service**.
   - Connect your backend repository.
   - Render will detect the `render.yaml` file (or select **Python** runtime).
3. **Environment Variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key.
   - `JWT_SECRET`: A strong random string for JWT signing.
   - `DATABASE_URL`: `sqlite:///./sql_app.db` (for free ephemeral storage) or your Render PostgreSQL URL.
   - `FRONTEND_URL`: The URL of your deployed Vercel frontend (e.g., `https://bi-copilot.vercel.app`).
   - `PYTHON_VERSION`: `3.11.0`

## Local Setup

1. Create a virtual environment: `python -m venv venv`
2. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in your keys.
5. Run: `uvicorn app.main:app --reload`
