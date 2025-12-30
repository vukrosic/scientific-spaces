import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scientific Spaces | Deep Learning & Mathematics",
  description: "Exploring the frontiers of mathematics, optimization, and large language models.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
