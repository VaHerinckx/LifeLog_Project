// src/components/charts/TopPodcastsChart/index.jsx
import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import _ from 'lodash';
import './TopPodcastsChart.css';

const formatNumber = (num) => {
  return new Intl.NumberFormat().format(Math.round(num));
};

const TopPodcastsChart = ({ data }) => {
  const [topPodcasts, setTopPodcasts] = useState([]);

  useEffect(() => {
    if (!Array.isArray(data)) return;

    const podcastStats = _(data)
      .groupBy('podcast_name')
      .map((episodes, name) => ({
        name,
        artwork: episodes[0].artwork_large,
        totalMinutes: _.sumBy(episodes, episode => {
          const duration = parseFloat(episode.duration);
          return isNaN(duration) ? 0 : duration / 60;
        })
      }))
      .orderBy(['totalMinutes'], ['desc'])
      .take(5)
      .value();

    setTopPodcasts(podcastStats);
  }, [data]);

  const CustomBar = (props) => {
    const { x, y, width, height, name, artwork } = props;
    return (
      <g>
        {/* Text label first (podcast name) */}
        <text
          x={0}
          y={y + height/2}
          dy=".35em"
          textAnchor="start"
          className="podcast-name"
        >
          {name}
        </text>
        {/* Artwork between text and bar */}
        <image
          x={x - 30} // Position artwork just before the bar
          y={y + (height - 20) / 2}
          href={artwork}
          className="podcast-artwork"
          preserveAspectRatio="xMidYMid slice"
        />
        {/* The bar itself */}
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill="#3423A6"
          opacity={0.8}
          rx={4}
        />
        {/* Value label at the end of the bar */}
        <text
          x={x + width + 5}
          y={y + height/2}
          dy=".35em"
          textAnchor="start"
          className="value-label"
        >
          {formatNumber(props.value)}m
        </text>
      </g>
    );
  };

  return (
    <div className="top-podcasts-container">
      <h3 className="top-podcasts-title">Top 5 Most Listened Podcasts</h3>
      <div className="top-podcasts-chart">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={topPodcasts}
            layout="vertical"
            margin={{ top: 20, right: 60, bottom: 20, left: 200 }} // Increased left margin for names
          >
            <XAxis
              type="number"
              tickFormatter={formatNumber}
              domain={[0, 'dataMax']}
            />
            <YAxis
              type="category"
              dataKey="name"
              hide // Hide default axis as we're rendering custom labels
            />
            <Bar
              dataKey="totalMinutes"
              shape={<CustomBar />}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TopPodcastsChart;
