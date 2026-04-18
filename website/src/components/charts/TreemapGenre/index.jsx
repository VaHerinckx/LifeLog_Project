import React, { useMemo } from 'react';
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts';
import _ from 'lodash';
import './TreemapGenre.css';

const COLORS = {
  base: '#3423A6',      // Darkest purple
  semidark: '#4433B6',  // Semi-dark purple
  medium1: '#5443C6',   // Medium purple 1
  medium2: '#6453D6',   // Medium purple 2
  medium3: '#7463E6',   // Medium purple 3
  light: '#8F84E8',     // Light purple
  stroke: '#171738',    // Stroke color
  text: '#FFFFFF'       // Text color
};

const TreemapGenre = ({ data, selectedPodcast, dateRange }) => {
  const treeMapData = useMemo(() => {
    if (!data || !dateRange.startDate || !dateRange.endDate) {
      return [];
    }

    // Filter data based on selection and date range
    const filteredData = data.filter(item => {
      const itemDate = new Date(item['modified at']);
      const start = new Date(dateRange.startDate);
      const end = new Date(dateRange.endDate);

      return (
        (selectedPodcast === 'all' || item.podcast_name === selectedPodcast) &&
        itemDate >= start &&
        itemDate <= end &&
        item.genre
      );
    });

    // Clean and normalize genres before grouping
    const cleanedData = filteredData.map(item => ({
      ...item,
      genre: item.genre?.includes('https://') || item.genre?.includes('image/thumb') ? 'Unknown' : item.genre
    }));

    // Group episodes by genre and calculate total listening time
    const genreGroups = _.groupBy(cleanedData, 'genre');

    // Convert to treemap format and calculate metrics
    const genreData = Object.entries(genreGroups)
      .map(([genre, episodes]) => {
        const totalMinutes = Math.round(
          episodes.reduce((sum, episode) => {
            const duration = parseFloat(episode.duration) || 0;
            return sum + (duration / 60);
          }, 0)
        );

        return {
          name: genre || 'Unknown',
          size: totalMinutes || 0
        };
      })
      .filter(item => item.size > 0) // Remove genres with no listening time
      .sort((a, b) => b.size - a.size); // Sort by size descending

    return genreData;
  }, [data, selectedPodcast, dateRange]);

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    return (
      <div className="treemap-tooltip">
        <p className="tooltip-title">{data.name}</p>
        <p className="tooltip-content">Total Time: {data.size.toLocaleString()} minutes</p>
      </div>
    );
  };

  // Improved CustomizedContent with text wrapping
  const CustomizedContent = ({ root, depth, x, y, width, height, name, value }) => {
    const maxSize = root.children?.[0]?.value || 0;
    const ratio = value / maxSize;

    let fill;
    if (ratio > 0.8) fill = COLORS.base;
    else if (ratio > 0.6) fill = COLORS.semidark;
    else if (ratio > 0.4) fill = COLORS.medium1;
    else if (ratio > 0.2) fill = COLORS.medium2;
    else if (ratio > 0.1) fill = COLORS.medium3;
    else fill = COLORS.light;

    // Only attempt to render text if we have enough space
    const shouldRenderText = width > 30 && height > 20;

    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={fill}
          stroke={COLORS.stroke}
          strokeWidth={1}
        />
        {shouldRenderText && (
          <foreignObject x={x} y={y} width={width} height={height}>
            <div
              xmlns="http://www.w3.org/1999/xhtml"
              style={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '4px',
                overflow: 'hidden',
                color: COLORS.text,
                fontSize: width < 80 ? '10px' : '14px',
                textAlign: 'center',
                wordBreak: 'break-word',
                textOverflow: 'ellipsis'
              }}
            >
              {name}
            </div>
          </foreignObject>
        )}
      </g>
    );
  };

  if (!treeMapData.length) {
    return (
      <div className="treemap-container">
        <h2 className="treemap-title">Genre Distribution</h2>
        <div className="treemap-chart-container">
          <p className="no-data-message">No genre data available for the selected filters.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="treemap-container">
      <h2 className="treemap-title">Genre Distribution</h2>
      <div className="treemap-chart-container">
        <ResponsiveContainer width="100%" height={400}>
          <Treemap
            data={treeMapData}
            dataKey="size"
            aspectRatio={4 / 3}
            stroke="light-white"
            fill="#3423A6"
            content={CustomizedContent}
          >
            <Tooltip content={CustomTooltip} />
          </Treemap>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TreemapGenre;
