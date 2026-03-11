## VisionFlow Demo Site

Interactive portfolio demo site for the **VisionFlow** traffic-vision and license plate recognition system, built with **Next.js (App Router)**, **React**, **TailwindCSS**, and **TypeScript** and ready for deployment on **Vercel**.

### 1. Installing dependencies

From the project root where `package.json` lives:

```bash
cd visionflow-demo-site
npm install
```

### 2. Running locally

Start the development server:

```bash
npm run dev
```

Then open `http://localhost:3000` in your browser.

Routes:

- `/` – Overview / landing page with hero, problem statement, pipeline and tech stack.
- `/demo` – Interactive demo with image upload or sample selection and simulated processing.
- `/results` – Results gallery with input/output examples.
- `/architecture` – Architecture and data flow explanation.

### 3. Deploying to Vercel

1. Push this project to GitHub (the `visionflow-demo-site` folder as its own project or as the root of a separate repo).
2. Go to the Vercel dashboard and create a **New Project**.
3. Import the GitHub repository that contains this Next.js app.
4. Ensure the **Framework Preset** is set to **Next.js**.
5. Keep the default build settings (`npm install`, `npm run build`, output directory `.next`).
6. Click **Deploy**. Vercel will build and host your demo site and provide a production URL.

