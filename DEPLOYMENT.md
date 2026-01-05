# Deployment Guide

## 🚀 Deploy to Vercel

### Prerequisites
- GitHub account
- Vercel account (sign up at [vercel.com](https://vercel.com))
- Git installed locally

### Step 1: Push to GitHub

1. **Initialize Git repository:**
```bash
git init
git add .
git commit -m "Initial commit: Support ticket classification system"
```

2. **Add remote and push:**
```bash
git remote add origin https://github.com/ijonathans/nlp-support-ticket.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Vercel

#### Option A: Deploy via Vercel Dashboard (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New Project"**
3. Import your GitHub repository: `ijonathans/nlp-support-ticket`
4. Vercel will auto-detect the Flask app
5. Configure settings:
   - **Framework Preset:** Other
   - **Build Command:** (leave empty)
   - **Output Directory:** (leave empty)
   - **Install Command:** `pip install -r requirements.txt`
6. Click **"Deploy"**

#### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI:**
```bash
npm install -g vercel
```

2. **Login to Vercel:**
```bash
vercel login
```

3. **Deploy:**
```bash
vercel
```

4. **Deploy to production:**
```bash
vercel --prod
```

### Step 3: Verify Deployment

Once deployed, Vercel will provide a URL like:
```
https://nlp-support-ticket.vercel.app
```

Visit the URL to test your app!

---

## ⚙️ Configuration Files

### `vercel.json`
Configures Vercel to run the Flask app:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### `requirements.txt`
Production dependencies (minimal for faster deployment):
```
flask==3.0.0
torch==2.1.0
numpy==1.24.3
```

---

## 🔧 Troubleshooting

### Issue: Deployment fails with "Module not found"
**Solution:** Make sure all dependencies are in `requirements.txt`

### Issue: Model files not loading
**Solution:** Ensure model files are committed to Git:
```bash
git add models/cnn_balanced.pt
git add artifacts/vocab_balanced.json
git add artifacts/label_map_balanced.json
git commit -m "Add model files"
git push
```

### Issue: Deployment timeout
**Solution:** PyTorch models can be large. Vercel has a 250MB limit for serverless functions. Consider:
- Using a smaller model
- Deploying to alternative platforms (Heroku, Railway, Render)

### Issue: Cold start latency
**Solution:** Vercel serverless functions have cold starts. First request may be slow (~5-10s). Subsequent requests are fast.

---

## 🌐 Alternative Deployment Options

### Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create nlp-support-ticket
git push heroku main
```

### Railway
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Flask and deploys

### Render
1. Go to [render.com](https://render.com)
2. Click "New Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`

---

## 📝 Post-Deployment

### Update README with Live URL
After deployment, update `README.md`:
```markdown
## 🌐 Live Demo
**Try it online:** [https://your-app-name.vercel.app](https://your-app-name.vercel.app)
```

### Monitor Performance
- Check Vercel dashboard for logs and analytics
- Monitor response times and errors
- Set up alerts for downtime

---

## 🔒 Security Notes

- Model files and artifacts are public in your repository
- No sensitive data (API keys, passwords) in code
- Use environment variables for secrets if needed
- Consider adding rate limiting for production use

---

## 📊 Performance Expectations

- **Cold Start:** 5-10 seconds (first request)
- **Warm Requests:** 100-500ms
- **Model Inference:** ~1.3ms per prediction
- **Concurrent Users:** Vercel scales automatically

---

## 🎉 Success!

Your support ticket classification system is now live and accessible worldwide!
