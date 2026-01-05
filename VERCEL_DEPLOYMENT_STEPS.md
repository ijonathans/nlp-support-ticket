# 🚀 Vercel Deployment - Step by Step

Your code is now on GitHub! Follow these steps to deploy on Vercel:

---

## ✅ Prerequisites Complete
- [x] Git repository initialized
- [x] Code pushed to GitHub: https://github.com/ijonathans/nlp-support-ticket.git
- [x] `vercel.json` configuration created
- [x] `requirements.txt` optimized for production

---

## 📋 Deployment Steps

### Step 1: Sign Up/Login to Vercel

1. Go to **[vercel.com](https://vercel.com)**
2. Click **"Sign Up"** or **"Login"**
3. Choose **"Continue with GitHub"**
4. Authorize Vercel to access your GitHub account

---

### Step 2: Import Your Project

1. On Vercel dashboard, click **"Add New..."** → **"Project"**
2. You'll see a list of your GitHub repositories
3. Find **`nlp-support-ticket`** and click **"Import"**

---

### Step 3: Configure Project Settings

Vercel will auto-detect your project. Configure these settings:

#### Project Settings:
- **Project Name:** `nlp-support-ticket` (or customize)
- **Framework Preset:** **Other** (or leave as detected)
- **Root Directory:** `./` (leave as default)

#### Build & Development Settings:
- **Build Command:** Leave empty (not needed for Flask)
- **Output Directory:** Leave empty
- **Install Command:** `pip install -r requirements.txt` (auto-detected)

#### Environment Variables:
- None required for this project (add if needed later)

---

### Step 4: Deploy!

1. Click **"Deploy"**
2. Vercel will:
   - Install dependencies from `requirements.txt`
   - Build your project
   - Deploy to a production URL
3. Wait 2-5 minutes for deployment to complete

---

### Step 5: Access Your App

Once deployed, you'll see:
```
✅ Deployment Complete!
🌐 https://nlp-support-ticket.vercel.app
```

Click the URL to test your app!

---

## 🎯 What to Test

1. **Home Page:** Should load the ticket classification interface
2. **Submit a Ticket:** Try these examples:
   - "My credit card was charged twice" → Should predict a department
   - "Problem with Account Charges" → Should predict Billing with high confidence
   - "Cannot login to my account" → Should predict Tech Support

3. **Check Confidence Scores:** Verify the confidence bars are displayed

---

## 🔧 Troubleshooting

### Issue: "Build Failed"
**Check:**
- Look at the build logs in Vercel dashboard
- Ensure `requirements.txt` has correct package versions
- Model files are committed to Git

### Issue: "500 Internal Server Error"
**Check:**
- Function logs in Vercel dashboard
- Model files (`cnn_balanced.pt`) are accessible
- Artifact files (`vocab_balanced.json`, `label_map_balanced.json`) exist

### Issue: "Module not found"
**Solution:**
- Add missing package to `requirements.txt`
- Redeploy from Vercel dashboard

### Issue: Deployment Timeout or Size Limit
**Note:** Vercel has a 250MB limit for serverless functions. PyTorch models are large.

**Alternative Solutions:**
1. **Use Vercel Edge Functions** (lighter runtime)
2. **Deploy to Railway:** [railway.app](https://railway.app) - No size limits
3. **Deploy to Render:** [render.com](https://render.com) - Free tier available
4. **Use Hugging Face Spaces:** [huggingface.co/spaces](https://huggingface.co/spaces)

---

## 🔄 Redeploy After Changes

Every time you push to GitHub, Vercel will automatically redeploy:

```bash
# Make changes to your code
git add .
git commit -m "Update model or fix bugs"
git push origin main
```

Vercel will detect the push and redeploy automatically!

---

## 📊 Monitor Your Deployment

### Vercel Dashboard Features:
- **Deployments:** View all deployment history
- **Analytics:** See page views and performance
- **Logs:** Debug errors and check function execution
- **Domains:** Add custom domain (optional)

---

## 🎉 Success Checklist

- [ ] Vercel account created
- [ ] Project imported from GitHub
- [ ] Deployment successful
- [ ] App accessible via Vercel URL
- [ ] Test predictions working
- [ ] Update README.md with live URL

---

## 📝 Update README with Live URL

After successful deployment, update your `README.md`:

```markdown
## 🌐 Live Demo
**Try it online:** [https://nlp-support-ticket.vercel.app](https://nlp-support-ticket.vercel.app)
```

Then commit and push:
```bash
git add README.md
git commit -m "Add live demo URL"
git push origin main
```

---

## 🌟 Optional Enhancements

### Add Custom Domain
1. Go to Vercel dashboard → Your project → Settings → Domains
2. Add your custom domain (e.g., `tickets.yourdomain.com`)
3. Follow DNS configuration instructions

### Enable Analytics
1. Go to Vercel dashboard → Your project → Analytics
2. Enable Web Analytics (free tier available)
3. Track visitors, page views, and performance

### Set Up Monitoring
1. Use Vercel's built-in monitoring
2. Or integrate with external services (Sentry, LogRocket)

---

## 🚨 Important Notes

1. **Cold Starts:** First request may take 5-10 seconds (Vercel serverless limitation)
2. **Concurrent Requests:** Vercel scales automatically
3. **Free Tier Limits:** 
   - 100GB bandwidth/month
   - 100 hours serverless function execution
   - Unlimited deployments

4. **Model Size:** If deployment fails due to size:
   - Consider model compression
   - Use alternative platforms (Railway, Render)
   - Split model loading across multiple functions

---

## 📞 Need Help?

- **Vercel Docs:** [vercel.com/docs](https://vercel.com/docs)
- **Vercel Support:** [vercel.com/support](https://vercel.com/support)
- **Community:** [github.com/vercel/vercel/discussions](https://github.com/vercel/vercel/discussions)

---

## ✅ You're All Set!

Your support ticket classification system is ready for the world! 🎉

**Next Steps:**
1. Deploy on Vercel following steps above
2. Test the live app
3. Share the URL with others
4. Monitor performance and iterate

Good luck! 🚀
