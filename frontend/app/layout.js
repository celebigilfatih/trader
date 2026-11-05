import "./globals.css"

export const metadata = {
  title: "Trader Cockpit",
  description: "BIST karar destek cockpit'i (API ile demo ön yüz)",
}

export default function RootLayout({ children }) {
  return (
    <html lang="tr">
      <body>{children}</body>
    </html>
  )
}
