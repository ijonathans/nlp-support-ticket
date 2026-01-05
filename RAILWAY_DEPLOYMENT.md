# 🚀 Railway Deployment Guide

Railway is perfect for ML/AI applications with PyTorch - no size limits and easy deployment!

---

## ✅ Why Railway?

- ✅ **No size limits** - Perfect for PyTorch models
- ✅ **Auto-detects Flask** - Zero configuration needed
- ✅ **Free tier available** - $5 free credit monthly
- ✅ **Auto-deploys from GitHub** - Push to deploy
- ✅ **Better for ML apps** - Optimized for larger dependencies

---

## 📋 Deployment Steps

### Step 1: Sign Up for Railway

1. Go to **[railway.app](https://railway.app)**
2. Click **"Login"** or **"Start a New Project"**
3. Choose **"Login with GitHub"**
4. Authorize Railway to access your GitHub account

---

### Step 2: Create New Project

1. On Railway dashboard, click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Find and select **`nlp-support-ticket`**
4. Railway will automatically:
   - Detect it's a Flask app
   - Install dependencies from `requirements.txt`
   - Use the `Procfile` to start the app
   - Deploy to a public URL

---

### Step 3: Wait for Deployment

1. Railway will show deployment progress
2. Wait 3-5 minutes for:
   - Installing PyTorch (~800MB)
   - Installing Flask and other dependencies
   - Starting the application
3. Watch the build logs in real-time

---

### Step 4: Access Your App

Once deployed, Railway provides a URL like:
```
https://nlp-support-ticket-production.up.railway.app
```

Click **"Generate Domain"** in Railway dashboard if not auto-generated.

---

## 🎯 What to Test

1. **Home Page:** Should load the ticket classification interface
2. **Submit Test Tickets:**
   - "My credit card was charged twice" → Customer Service
   - "Problem with Account Charges" → Billing (high confidence)
   - "Cannot login to my account" → Tech Support

3. **Check Confidence Scores:** Verify bars display correctly

---

## 📊 Railway Dashboard Features

### Deployments Tab
- View all deployment history
- See build logs
- Check deployment status

### Metrics Tab
- CPU usage
- Memory usage
- Network traffic

### Variables Tab
- Add environment variables if needed
- Currently none required for this project

### Settings Tab
- Custom domain configuration
- Deployment settings
- Delete project

---

## 🔄 Auto-Deploy on Git Push

Every time you push to GitHub, Railway automatically redeploys:

```bash
# Make changes to your code
git add .
git commit -m "Update model or fix bugs"
git push origin main
```

Railway detects the push and redeploys automatically!

---

## 💰 Pricing & Free Tier

### Free Tier (Hobby Plan)
- **$5 free credit per month**
- ~500 hours of runtime
- Perfect for personal projects and demos
- No credit card required to start

### Usage Estimates
- **Idle:** ~$0.01/hour
- **Active:** ~$0.02/hour
- **Monthly (24/7):** ~$10-15

**Tip:** Railway sleeps after inactivity to save credits!

---

## ⚙️ Configuration Files

### `Procfile`
Tells Railway how to start the app:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### `requirements.txt`
Dependencies Railway will install:
```
flask==3.0.0
torch==2.1.2
numpy==1.24.3
gunicorn==21.2.0
```

---

## 🔧 Troubleshooting

### Issue: "Build Failed"
**Check:**
- Look at build logs in Railway dashboard
- Ensure `requirements.txt` has correct versions
- Model files are committed to Git

### Issue: "Application Error" or 502
**Check:**
- Deployment logs in Railway dashboard
- Model files (`cnn_balanced.pt`) are accessible
- Artifact files exist in repository

### Issue: "Out of Memory"
**Solution:**
- Railway provides 8GB RAM by default
- PyTorch models should fit comfortably
- If needed, upgrade to higher plan

### Issue: Slow First Load
**Normal:** First request loads the model into memory (~2-5 seconds)
**Subsequent requests:** Fast (~100-200ms)

---

## 🌟 Optional Enhancements

### Add Custom Domain

1. Go to Railway dashboard → Your project → Settings
2. Click **"Generate Domain"** or **"Custom Domain"**
3. Add your custom domain (e.g., `tickets.yourdomain.com`)
4. Follow DNS configuration instructions

### Set Up Environment Variables

If you need API keys or secrets:
1. Go to Variables tab
2. Click **"New Variable"**
3. Add key-value pairs
4. Redeploy automatically applies changes

### Enable Monitoring

Railway provides built-in monitoring:
- CPU and memory usage graphs
- Request logs
- Error tracking
- Performance metrics

---

## 📝 Update README with Live URL

After successful deployment, update your `README.md`:

```markdown
## 🌐 Live Demo
**Try it online:** [https://your-app.up.railway.app](https://your-app.up.railway.app)
```

Then commit and push:
```bash
git add README.md
git commit -m "Add Railway deployment URL"
git push origin main
```

---

## 🚀 Advantages Over Other Platforms

| Feature | Railway | Vercel | Heroku |
|---------|---------|--------|--------|
| **PyTorch Support** | ✅ Excellent | ⚠️ Size limits | ✅ Good |
| **Auto-deploy** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Free Tier** | ✅ $5/month | ✅ Limited | ❌ Paid only |
| **Setup Complexity** | ✅ Zero config | ⚠️ Config needed | ⚠️ Config needed |
| **ML/AI Optimized** | ✅ Yes | ❌ No | ⚠️ Partial |
| **Size Limits** | ✅ None | ❌ 250MB | ✅ 500MB |

---

## 🎉 Success Checklist

- [ ] Railway account created
- [ ] Project deployed from GitHub
- [ ] Deployment successful (check logs)
- [ ] App accessible via Railway URL
- [ ] Test predictions working
- [ ] Custom domain added (optional)
- [ ] README updated with live URL

---

## 📞 Need Help?

- **Railway Docs:** [docs.railway.app](https://docs.railway.app)
- **Railway Discord:** [discord.gg/railway](https://discord.gg/railway)
- **Community Forum:** [help.railway.app](https://help.railway.app)

---

## ✅ You're All Set!

Railway is the perfect platform for your ML-powered support ticket classifier! 🎉

**Next Steps:**
1. Deploy on Railway (takes 5 minutes)
2. Test the live app
3. Share your URL
4. Monitor usage in dashboard

**Your app will be live at:**
```
https://nlp-support-ticket-production.up.railway.app
```

Good luck! 🚀
