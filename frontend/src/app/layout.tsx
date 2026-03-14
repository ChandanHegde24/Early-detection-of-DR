import "./globals.css";
import "@/styles/tokens.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Diabetic Retinopathy Screening",
  description: "Frontend for DR unified screening workflow",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
