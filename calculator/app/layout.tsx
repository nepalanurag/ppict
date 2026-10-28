import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ppict calculator",
  description:
    "Prediction-powered covariate adjustment for randomized trials, running the ppict Python estimator on your data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
