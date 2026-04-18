import { useMemo } from 'react';
import { X, Activity, Moon, Heart, MapPin, Smartphone, Brain, StickyNote, Dumbbell, Home, Clock } from 'lucide-react';
import './HealthDetails.css';
import { StarRating } from '../../../components/ui';
import { formatDate } from '../../../utils';

/**
 * Get the appropriate emoji for an evaluation score (1-5)
 * @param {number} rating - The evaluation score
 * @returns {string} Emoji string
 */
const getEvaluationEmoji = (rating) => {
  if (!rating || rating === 0) return '💪'; // Default health emoji
  const score = Math.round(rating);

  const emojiMap = {
    5: '😄',  // Very happy
    4: '🙂',  // Happy
    3: '😐',  // Neutral
    2: '😕',  // Unhappy
    1: '😞',  // Very unhappy
  };
  return emojiMap[score] || '💪';
};

/**
 * Create a chronological timeline of the day from hourly segment records
 * Groups by segment_id, includes both stationary and moving segments, sorted by start time
 */
const createDayTimeline = (hourlyData) => {
  if (!hourlyData || hourlyData.length === 0) return [];

  const segmentMap = new Map(); // segment_id -> segment data

  hourlyData.forEach(segment => {
    const segmentId = segment.segment_id;
    if (!segmentId) return;

    // Skip gap-fill segments (they have '_gap' in segment_id)
    if (segmentId.includes('_gap')) return;

    if (!segmentMap.has(segmentId)) {
      segmentMap.set(segmentId, {
        segment_id: segmentId,
        segment_type: segment.segment_type,
        segment_start_time: segment.segment_start_time,
        segment_end_time: segment.segment_end_time,
        place_name: segment.place_name,
        address: segment.address,
        city: segment.city,
        country: segment.country,
        is_home: segment.is_home,
        activity_type: segment.activity_type,
        totalMinutes: 0,
        totalDistance: 0
      });
    }

    const entry = segmentMap.get(segmentId);
    entry.totalMinutes += segment.segment_duration_minutes || 0;
    entry.totalDistance += segment.distance_meters || 0;
  });

  // Convert to array and sort by start time ascending (chronological order)
  return Array.from(segmentMap.values())
    .filter(seg => seg.segment_start_time) // Only include segments with start time
    .sort((a, b) => a.segment_start_time.localeCompare(b.segment_start_time));
};

/**
 * Aggregate activity data from hourly segment records
 * Groups by activity type, sums minutes and distance
 */
const aggregateActivities = (hourlyData) => {
  if (!hourlyData || hourlyData.length === 0) return [];

  const activityMap = new Map(); // activity_type -> { minutes, distance_m }

  hourlyData.forEach(segment => {
    const activityType = segment.activity_type;
    if (!activityType || activityType === 'idle') return; // Skip idle/stationary

    if (!activityMap.has(activityType)) {
      activityMap.set(activityType, { minutes: 0, distance_m: 0 });
    }

    const entry = activityMap.get(activityType);
    entry.minutes += segment.segment_duration_minutes || 0;
    entry.distance_m += segment.distance_meters || 0;
  });

  // Convert to array and sort by time (most to least)
  return Array.from(activityMap.entries())
    .map(([type, data]) => ({
      type,
      displayName: type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' '),
      minutes: data.minutes,
      distance_m: data.distance_m
    }))
    .sort((a, b) => b.minutes - a.minutes);
};

/**
 * Aggregate hourly metrics into daily totals
 */
const aggregateHourlyMetrics = (hourlyData) => {
  if (!hourlyData || hourlyData.length === 0) return null;

  const totals = {
    steps: 0,
    active_energy_kcal: 0,
    resting_energy_kcal: 0,
    screen_time_minutes: 0,
    phone_pickups: 0,
    distance_meters: 0,
    apple_distance_meters: 0,
    flights_climbed: 0,
    avg_heart_rate: 0,
    avg_heart_rate_count: 0,
    body_weight_kg: null,
    body_fat_percent: null
  };

  hourlyData.forEach(segment => {
    totals.steps += segment.steps || 0;
    totals.active_energy_kcal += segment.active_energy_kcal || 0;
    totals.resting_energy_kcal += segment.resting_energy_kcal || 0;
    totals.screen_time_minutes += segment.screen_time_minutes || 0;
    totals.phone_pickups += segment.phone_pickups || 0;
    totals.distance_meters += segment.distance_meters || 0;
    totals.apple_distance_meters += segment.apple_distance_meters || 0;
    totals.flights_climbed += segment.flights_climbed || 0;

    if (segment.avg_heart_rate && segment.avg_heart_rate > 0) {
      totals.avg_heart_rate += segment.avg_heart_rate;
      totals.avg_heart_rate_count++;
    }

    // Take most recent body metrics
    if (segment.body_weight_kg && segment.body_weight_kg > 0) {
      totals.body_weight_kg = segment.body_weight_kg;
    }
    if (segment.body_fat_percent && segment.body_fat_percent > 0) {
      totals.body_fat_percent = segment.body_fat_percent;
    }
  });

  // Calculate average heart rate
  if (totals.avg_heart_rate_count > 0) {
    totals.avg_heart_rate = totals.avg_heart_rate / totals.avg_heart_rate_count;
  }

  return totals;
};

const HealthDetails = ({ day, hourlyData = [], onClose }) => {
  // Debug: log raw data
  console.log('DEBUG hourlyData:', hourlyData.length, hourlyData[0]);

  // Aggregate location and activity data from hourly records
  // These must be called before any conditional returns (React hooks rules)
  const dayTimeline = useMemo(() => {
    const result = createDayTimeline(hourlyData);
    console.log('DEBUG dayTimeline:', result.length, result);
    return result;
  }, [hourlyData]);
  const aggregatedActivities = useMemo(() => aggregateActivities(hourlyData), [hourlyData]);
  const hourlyMetrics = useMemo(() => aggregateHourlyMetrics(hourlyData), [hourlyData]);

  if (!day) return null;

  const healthEmoji = getEvaluationEmoji(day.overall_evaluation);

  // Format helper functions
  const formatSteps = (steps) => {
    if (!steps || steps === 0) return 'No data';
    return `${Math.round(steps).toLocaleString()} steps`;
  };

  const formatDistance = (meters) => {
    if (!meters || meters === 0) return 'No data';
    return `${(meters / 1000).toFixed(2)} km`;
  };

  const formatEnergy = (kcal) => {
    if (!kcal || kcal === 0) return 'No data';
    return `${Math.round(kcal)} kcal`;
  };

  const formatMinutes = (minutes) => {
    if (!minutes || minutes === 0) return 'No data';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    if (hours === 0) return `${mins}m`;
    if (mins === 0) return `${hours}h 0m`;
    return `${hours}h ${mins}m`;
  };

  const formatHeartRate = (hr) => {
    if (!hr || hr === 0) return 'No data';
    return `${Math.round(hr)} bpm`;
  };

  const formatWeight = (weight) => {
    if (!weight || weight === 0) return 'No data';
    return `${weight.toFixed(1)} kg`;
  };

  const formatBodyFat = (fat) => {
    if (!fat || fat === 0) return 'No data';
    return `${(fat * 100).toFixed(1)}%`;
  };

  const formatScreenTime = (minutes) => {
    if (!minutes || minutes === 0) return 'No data';
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  return (
    <div className="health-details-overlay" onClick={onClose}>
      <div className="health-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="health-details-content">
          <div className="health-details-header">
            <div className="health-icon-large">
              <span className="health-emoji-large">{healthEmoji}</span>
            </div>
            <div className="health-header-info">
              <h2>{formatDate(day.date)}</h2>
              {day.sleep_start_time && day.wake_up_time && (
                <p className="health-sleep-times">
                  <Clock size={16} /> Sleep: {day.sleep_start_time} - {day.wake_up_time}
                </p>
              )}
            </div>
          </div>

          <div className="health-details-body">

            {/* Subjective Evaluations Section */}
            <div className="health-section">
              <h3 className="section-title">
                <Dumbbell size={20} />
                Evaluations
              </h3>
              <div className="section-content">
                {day.overall_evaluation && day.overall_evaluation > 0 && (
                  <div className="rating-item">
                    <label>Overall Feeling:</label>
                    <div className="rating-display">
                      <StarRating rating={day.overall_evaluation} maxRating={5} size={18} />
                      <span>{day.overall_evaluation.toFixed(1)}</span>
                    </div>
                  </div>
                )}
                {day.sleep_quality && day.sleep_quality > 0 && (
                  <div className="rating-item">
                    <label>Sleep Quality:</label>
                    <div className="rating-display">
                      <StarRating rating={day.sleep_quality} maxRating={5} size={18} />
                      <span>{day.sleep_quality.toFixed(1)} ({day.sleep_quality_text || 'N/A'})</span>
                    </div>
                  </div>
                )}
                {day.fitness_feeling && day.fitness_feeling > 0 && (
                  <div className="rating-item">
                    <label>Fitness Feeling:</label>
                    <div className="rating-display">
                      <StarRating rating={day.fitness_feeling} maxRating={5} size={18} />
                      <span>{day.fitness_feeling.toFixed(1)}</span>
                    </div>
                  </div>
                )}
                {day.sleep_rest_feeling && day.sleep_rest_feeling > 0 && (
                  <div className="rating-item">
                    <label>Rest Feeling:</label>
                    <div className="rating-display">
                      <StarRating rating={day.sleep_rest_feeling} maxRating={5} size={18} />
                      <span>{day.sleep_rest_feeling.toFixed(1)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Subjective Notes Section */}
            {(day.notes_summary || day.dream_description) && (
              <div className="health-section health-notes-section">
                <h3 className="section-title">
                  <StickyNote size={20} />
                  Notes & Reflections
                </h3>
                <div className="section-content">
                  {day.dream_description && (
                    <div className="note-item">
                      <div className="note-header">
                        <Brain size={16} />
                        <label>Dream ({day.dreams || 1} recorded):</label>
                      </div>
                      <p className="note-text">{day.dream_description}</p>
                    </div>
                  )}
                  {day.notes_summary && (
                    <div className="note-item">
                      <div className="note-header">
                        <StickyNote size={16} />
                        <label>Daily Notes:</label>
                      </div>
                      <p className="note-text">{day.notes_summary}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Activity Metrics Section - from hourly aggregation */}
            <div className="health-section">
              <h3 className="section-title">
                <Activity size={20} />
                Activity Metrics
              </h3>
              <div className="section-content">
                <div className="detail-item">
                  <label>Steps:</label>
                  <span>{formatSteps(day.total_steps || hourlyMetrics?.steps)}</span>
                </div>
                <div className="detail-item">
                  <label>Walking Distance:</label>
                  <span>{formatDistance(day.total_apple_distance_meters || hourlyMetrics?.apple_distance_meters)}</span>
                </div>
                <div className="detail-item">
                  <label>Flights Climbed:</label>
                  <span>{(day.total_flights_climbed || hourlyMetrics?.flights_climbed) || 'No data'}</span>
                </div>
                <div className="detail-item">
                  <label>Resting Energy:</label>
                  <span>{formatEnergy(day.total_resting_energy_kcal || hourlyMetrics?.resting_energy_kcal)}</span>
                </div>
                <div className="detail-item">
                  <label>Active Energy:</label>
                  <span>{formatEnergy(day.total_active_energy_kcal || hourlyMetrics?.active_energy_kcal)}</span>
                </div>
              </div>
            </div>

            {/* Sleep Metrics Section */}
            <div className="health-section">
              <h3 className="section-title">
                <Moon size={20} />
                Sleep Metrics
              </h3>
              <div className="section-content">
                <div className="detail-item">
                  <label>Total Sleep:</label>
                  <span>{formatMinutes(day.total_sleep_minutes)}</span>
                </div>
                <div className="detail-item">
                  <label>Core Sleep:</label>
                  <span>{formatMinutes(day.total_core_sleep_minutes)}</span>
                </div>
                <div className="detail-item">
                  <label>Deep Sleep:</label>
                  <span>{formatMinutes(day.total_deep_sleep_minutes)}</span>
                </div>
                <div className="detail-item">
                  <label>REM Sleep:</label>
                  <span>{formatMinutes(day.total_rem_sleep_minutes)}</span>
                </div>
                <div className="detail-item">
                  <label>Time Awake:</label>
                  <span>{formatMinutes(day.total_awake_minutes)}</span>
                </div>
                {day.sleep_start_time && day.wake_up_time && (
                  <div className="detail-item">
                    <label>Sleep Schedule:</label>
                    <span>{day.sleep_start_time} - {day.wake_up_time}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Body Metrics Section - from hourly aggregation */}
            <div className="health-section">
              <h3 className="section-title">
                <Heart size={20} />
                Body Metrics
              </h3>
              <div className="section-content">
                <div className="detail-item">
                  <label>Avg Heart Rate:</label>
                  <span>{formatHeartRate(hourlyMetrics?.avg_heart_rate)}</span>
                </div>
                <div className="detail-item">
                  <label>Body Weight:</label>
                  <span>{formatWeight(hourlyMetrics?.body_weight_kg)}</span>
                </div>
                <div className="detail-item">
                  <label>Body Fat:</label>
                  <span>{formatBodyFat(hourlyMetrics?.body_fat_percent)}</span>
                </div>
              </div>
            </div>

            {/* Day Timeline Section */}
            {dayTimeline.length > 0 && (
              <div className="health-section">
                <h3 className="section-title">
                  <MapPin size={20} />
                  Day Timeline
                </h3>
                <div className="section-content">
                  <div className="breakdown-list">
                    {dayTimeline.map((seg, index) => (
                      <div key={index} className="breakdown-item">
                        <div className="breakdown-item-main">
                          <span className="breakdown-name">
                            {seg.segment_type === 'stationary'
                              ? (seg.place_name || seg.city || 'Unknown Location')
                              : (seg.activity_type ? seg.activity_type.charAt(0).toUpperCase() + seg.activity_type.slice(1).replace(/_/g, ' ') : 'Moving')
                            }
                            {seg.is_home && (
                              <span className="home-badge">
                                <Home size={12} />
                                Home
                              </span>
                            )}
                          </span>
                          <span className="breakdown-value">{formatMinutes(seg.totalMinutes)}</span>
                        </div>
                        <div className="breakdown-item-sub">
                          {seg.segment_type === 'stationary' && seg.city && seg.place_name && (
                            <span className="breakdown-city">{seg.city}</span>
                          )}
                          {seg.segment_type === 'moving' && seg.totalDistance > 0 && (
                            <span className="breakdown-distance">{(seg.totalDistance / 1000).toFixed(1)} km</span>
                          )}
                        </div>
                        <div className="breakdown-item-time">
                          <Clock size={12} />
                          <span>{seg.segment_start_time} - {seg.segment_end_time}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Activity Breakdown Section */}
            {aggregatedActivities.length > 0 && (
              <div className="health-section">
                <h3 className="section-title">
                  <Activity size={20} />
                  Activity Breakdown
                </h3>
                <div className="section-content">
                  <div className="breakdown-list">
                    {aggregatedActivities.map((activity, index) => (
                      <div key={index} className="breakdown-item">
                        <div className="breakdown-item-main">
                          <span className="breakdown-name">{activity.displayName}</span>
                          <span className="breakdown-value">
                            {formatMinutes(activity.minutes)}
                            {activity.distance_m > 0 && (
                              <span className="breakdown-distance">
                                ({(activity.distance_m / 1000).toFixed(1)} km)
                              </span>
                            )}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Screen Time Section */}
            <div className="health-section">
              <h3 className="section-title">
                <Smartphone size={20} />
                Screen Time
              </h3>
              <div className="section-content">
                <div className="detail-item">
                  <label>Total Screen Time:</label>
                  <span>{formatScreenTime(day.total_screen_time_minutes || hourlyMetrics?.screen_time_minutes)}</span>
                </div>
                <div className="detail-item">
                  <label>Pickups:</label>
                  <span>{(day.total_phone_pickups || hourlyMetrics?.phone_pickups) || 'No data'}</span>
                </div>
                <div className="detail-item">
                  <label>Before Sleep:</label>
                  <span>{formatScreenTime(day.total_screen_time_minutes_before_sleep)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HealthDetails;
