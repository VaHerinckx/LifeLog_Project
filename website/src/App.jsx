// src/App.jsx
import './styles/variables.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { DataProvider } from './context/DataContext';
import { AuthProvider } from './context/AuthContext';
import { Layout } from './components/ui';
import Homepage from './pages/Home/Homepage';
import ReadingPage from './pages/Reading/ReadingPage';
import MoviesPage from './pages/Movies/MoviesPage';
import MusicPage from './pages/Music/MusicPage';
import NutritionPage from './pages/Nutrition/NutritionPage';
import PodcastPage from './pages/Podcast/PodcastPage';
import ShowsPage from './pages/Shows/ShowsPage';
import FinancePage from './pages/Finance/FinancePage';
import HealthPage from './pages/Health/HealthPage';

const App = () => {
  return (
    <AuthProvider>
    <DataProvider>
      <Router
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true
        }}
      >
        <Routes>
          {/* Homepage outside of Layout */}
          <Route path="/" element={<Homepage />} />

          {/* All other pages wrapped in Layout */}
          <Route element={<Layout />}>
            <Route path="/reading" element={<ReadingPage />} />
            <Route path="/movies" element={<MoviesPage />} />
            <Route path="/music" element={<MusicPage />} />
            <Route path="/nutrition" element={<NutritionPage />} />
            <Route path="/podcasts" element={<PodcastPage />} />
            <Route path="/shows" element={<ShowsPage />} />
            <Route path="/finance" element={<FinancePage />} />
            <Route path="/health" element={<HealthPage />} />
            <Route
              path="/sport"
              element={
                <div className="page-container">
                  <div className="coming-soon-text">Sport Page Coming Soon</div>
                </div>
              }
            />
            <Route
              path="/work"
              element={
                <div className="page-container">
                  <div className="coming-soon-text">Work Page Coming Soon</div>
                </div>
              }
            />
          </Route>
        </Routes>
      </Router>
      {/* Sticky bottom spacer for persistent viewport padding */}
      <div className="bottom-spacer" />
    </DataProvider>
    </AuthProvider>
  );
};

export default App;
