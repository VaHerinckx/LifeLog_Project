module.exports = {
  root: true,
  env: {
    browser: true,
    es2020: true,
    node: true
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs'],
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true
    }
  },
  settings: {
    react: {
      version: '18.2'
    }
  },
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExports: true },
    ],
    // Turn off or warn for common development rules
    'no-unused-vars': 'warn',
    'react/prop-types': 'off',
    'react-hooks/exhaustive-deps': 'warn',
    'no-console': 'warn',
    'react/react-in-jsx-scope': 'off',
    'react/display-name': 'off'
  }
}
