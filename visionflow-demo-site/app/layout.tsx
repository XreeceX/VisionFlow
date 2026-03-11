import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";

export const metadata: Metadata = {
  title: "VisionFlow – Intelligent Traffic Vision Demo",
  description:
    "Interactive portfolio demo for VisionFlow, a computer-vision powered traffic understanding and license plate recognition system."
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-background text-foreground antialiased">
        <div className="min-h-screen flex flex-col bg-[radial-gradient(circle_at_top,_#1f2937_0,_#020617_45%,_#000000_100%)]">
          <Navbar />
          <main className="flex-1 px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto w-full">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  );
}

