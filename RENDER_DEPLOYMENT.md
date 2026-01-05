# 🚀 Deploy to Render

Render is better for ML/AI apps with PyTorch - no image size limits on free tier!

---

## ✅ Why Render?

- ✅ **No image size limits** - Perfect for PyTorch
- ✅ **Free tier available** - 750 hours/month
- ✅ **Auto-deploys from GitHub** - Push to deploy
- ✅ **Better for ML apps** - Optimized for larger dependencies
- ✅ **Simple setup** - Zero configuration needed

---

## 📋 Quick Deploy Steps

### Step 1: Go to Render
Visit: **[render.com](https://render.com)**

### Step 2: Sign Up/Login
Click **"Get Started"** or **"Login"**  
Choose **"Sign in with GitHub"**

### Step 3: Create New Web Service
1. Click **"New +"** → **"Web Service"**
2. Click **"Connect account"** to link GitHub
3. Find and select **`nlp-support-ticket`** repository
4. Click **"Connect"**

### Step 4: Configure Service
Render will auto-detect settings, but verify:

- **Name:** `nlp-support-ticket`
- **Region:** Choose closest to you
- **Branch:** `main`
- **Runtime:** `Python 3`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Instance Type:** `Free`

### Step 5: Deploy!
1. Click **"Create Web Service"**
2. Wait 5-10 minutes for deployment
3. Watch the build logs

---

## 🌐 Your App URL

After deployment:
```
https://nlp-support-ticket.onrender.com
```

---

## 🧪 Test Your App

1. **"My credit card was charged twice"** → Customer Service
2. **"Problem with Account Charges"** → Billing (high confidence)
3. **"Cannot login to my account"** → Tech Support

---

## 💰 Render Free Tier

- **750 hours/month** - ~31 days of runtime
- **512 MB RAM** - Enough for this app
- **No credit card required**
- **Auto-sleep after 15 min inactivity** (saves hours)
- **First request after sleep: ~30 seconds** (cold start)

---

## 🔄 Auto-Deploy

Every GitHub push triggers automatic redeployment:

```bash
git add .
git commit -m "Update app"
git push origin main
# Render auto-deploys!
```

---

## 📊 Render Dashboard

- **Logs:** Real-time deployment and application logs
- **Metrics:** CPU, memory, request stats
- **Environment:** Add environment variables
- **Settings:** Custom domain, auto-deploy settings

---

## 🎉 Advantages Over Railway

| Feature | Render | Railway |
|---------|--------|---------|
| **Image Size Limit** | ✅ None | ❌ 4.8 GB (free tier) |
| **Free Hours** | ✅ 750/month | ⚠️ ~500/month |
| **Setup** | ✅ Simple | ✅ Simple |
| **PyTorch Support** | ✅ Excellent | ⚠️ Size issues |
| **Cold Start** | ⚠️ ~30s | ✅ ~5s |

---

## 🔧 Troubleshooting

### Issue: Build takes long time
**Normal:** PyTorch installation takes 5-10 minutes on first deploy

### Issue: "Out of Memory"
**Solution:** Render free tier has 512MB RAM. This should be enough, but if issues:
- Upgrade to paid tier ($7/month for 2GB RAM)
- Or use Hugging Face Spaces (free, unlimited)

### Issue: App sleeps after inactivity
**Normal:** Free tier sleeps after 15 minutes
**First request after sleep:** ~30 seconds to wake up
**Solution:** Upgrade to paid tier for always-on

---

## 📝 After Deployment

Update your README:
```markdown
## 🌐 Live Demo
**Try it online:** [https://nlp-support-ticket.onrender.com](https://nlp-support-ticket.onrender.com)
```

---

## ✅ You're All Set!

Render is perfect for your PyTorch app - no size limits! 🎉

**Deploy now:** [render.com](https://render.com) 🚀
