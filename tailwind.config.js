/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,js}"],
  theme: {
    extend: {
      colors: {
        'forex-green': '#10B981',
        'forex-red': '#EF4444',
        'forex-blue': '#3B82F6',
        'forex-yellow': '#F59E0B',
        'forex-dark': '#1F2937',
        'forex-darker': '#111827'
      }
    },
  },
  plugins: [],
}