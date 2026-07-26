/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0d12",
          900: "#12151d",
          850: "#171b25",
          800: "#1d2230",
          700: "#2a3142",
          600: "#3a4358",
        },
        accent: {
          DEFAULT: "#6366f1",
          hover: "#818cf8",
        },
      },
    },
  },
  plugins: [],
};
