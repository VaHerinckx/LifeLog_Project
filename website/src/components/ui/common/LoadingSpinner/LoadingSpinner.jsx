// LoadingSpinner.jsx
import { Activity, Music, Podcast, Book, Film, DollarSign, Brain, Utensils, Briefcase } from 'lucide-react';
import './LoadingSpinner.css';

const getLoadingMessage = (centerIcon) => {
  switch (centerIcon) {
    case Podcast:
      return "Loading your podcast listening history...";
    case Music:
      return "Tuning into your music data...";
    case Utensils:
      return "Gathering your nutrition information...";
    case Activity:
      return "Tracking down your sports activities...";
    case Brain:
      return "Analyzing your health metrics...";
    case Book:
      return "Flipping through your reading history...";
    case Film:
      return "Rolling your viewing history...";
    case DollarSign:
      return "Calculating your financial data...";
    case Briefcase:
      return "Compiling your work statistics...";
    default:
      return "Collecting your life data...";
  }
};

const LoadingSpinner = ({ centerIcon: CenterIcon = Podcast, containerClass = "", fullPage = true }) => {
  const loadingMessage = getLoadingMessage(CenterIcon);

  return (
    <div className={`${fullPage ? 'loading-container' : 'loading-container-inline'} ${containerClass}`}>
      <div className="solar-system">
        {/* Center "sun" icon */}
        <div className="center-icon">
          <CenterIcon size={48} />
        </div>

        {/* Inner orbit */}
        <div className="orbit orbit-inner">
          <div className="icon-container">
            <Activity size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <Film size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <DollarSign size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <Briefcase size={32} className="orbit-icon" />
          </div>
        </div>

        {/* Outer orbit */}
        <div className="orbit orbit-outer">
          <div className="icon-container">
            <Music size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <Brain size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <Book size={32} className="orbit-icon" />
          </div>
          <div className="icon-container">
            <Utensils size={32} className="orbit-icon" />
          </div>
        </div>
      </div>

      <div className="loading-message">
        <p className="loading-text">{loadingMessage}</p>
        <p className="loading-subtext">Please wait while we organize your personal insights...</p>
      </div>
    </div>
  );
};

export default LoadingSpinner;
