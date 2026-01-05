# Deployment Guide

## 🚀 Deploy to Railway (Recommended)

### Prerequisites
- GitHub account
- Railway account (sign up at [railway.app](https://railway.app))
- Git installed locally (already done!)

### Step 1: Code Already on GitHub ✅

Your code is already pushed to:
```
https://github.com/ijonathans/nlp-support-ticket.git
```

### Step 2: Deploy on Railway

1. Go to **[railway.app](https://railway.app)** and sign in with GitHub
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose **`nlp-support-ticket`** from the list
5. Railway will automatically:
   - Detect it's a Flask app
   - Install dependencies from `requirements.txt`
   - Use `Procfile` to start with Gunicorn
   - Deploy to production
6. Wait 3-5 minutes for deployment

### Step 3: Generate Domain

1. In Railway dashboard, go to your project
2. Click **"Settings"** tab
3. Scroll to **"Domains"**
4. Click **"Generate Domain"**
5. You'll get a URL like:
```
https://nlp-support-ticket-production.up.railway.app
```

Visit the URL to test your app!

---

## ⚙️ Configuration Files

### `Procfile`
Tells Railway how to start the app:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### `requirements.txt`
Production dependencies:
```
flask==3.0.0
torch==2.1.2
numpy==1.24.3
gunicorn==21.2.0
```

---

## 🔧 Troubleshooting

### Issue: Build takes a long time
**Normal:** PyTorch installation takes 2-3 minutes. Be patient!

### Issue: "Module not found"
**Solution:** Make sure all dependencies are in `requirements.txt`

### Issue: Model files not loading
**Solution:** Ensure model files are committed to Git (already done!)

### Issue: Application Error (502)
**Check:**
- Deployment logs in Railway dashboard
- Ensure `Procfile` is correct
- Verify model files exist in repository

### Issue: Out of memory
**Solution:** Railway provides 8GB RAM by default, which is plenty for this app. If issues persist, check for memory leaks.

---

## 🌐 Alternative Deployment Options

### Render
1. Go to [render.com](https://render.com)
2. Click "New Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

### Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create nlp-support-ticket
git push heroku main
```


---

## 📝 Post-Deployment

### Update README with Live URL
After deployment, update `README.md`:
```markdown
## 🌐 Live Demo
**Try it online:** [https://your-app.up.railway.app](https://your-app.up.railway.app)
```

### Monitor Performance
- Check Railway dashboard for logs and metrics
- Monitor CPU, memory, and network usage
- View deployment history

---

## 🔒 Security Notes

- Model files and artifacts are public in your repository
- No sensitive data (API keys, passwords) in code
- Use environment variables for secrets if needed
- Consider adding rate limiting for production use

---

## 📊 Performance Expectations

- **First Load:** 2-5 seconds (model loading)
- **Subsequent Requests:** 100-200ms
- **Model Inference:** ~1.3ms per prediction
- **Concurrent Users:** Railway handles multiple users well
- **Memory Usage:** ~500MB-1GB (PyTorch model)

---

## 🎉 Success!

Your support ticket classification system is now live and accessible worldwide!
