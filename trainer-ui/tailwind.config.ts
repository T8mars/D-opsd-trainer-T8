import type { Config } from 'tailwindcss';

const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#07080a',
          900: '#0d1014',
          850: '#12161c',
          800: '#171d25',
          700: '#26303b',
          600: '#3b4652',
          400: '#9aa7b3',
          300: '#c4ccd4',
          100: '#f2f6f9',
        },
        aqua: {
          500: '#45d3c6',
          300: '#8ee8df',
        },
        mint: {
          500: '#7adf9b',
        },
        amberSoft: {
          500: '#e9b85d',
        },
        roseSoft: {
          500: '#eb758a',
        },
      },
      boxShadow: {
        glass: '0 18px 60px rgba(0, 0, 0, 0.34)',
        insetGlass: 'inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -1px 0 rgba(255,255,255,0.06)',
      },
    },
  },
  plugins: [],
};

export default config;
