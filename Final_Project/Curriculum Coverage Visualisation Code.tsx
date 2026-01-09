// CurriculumCoverage.tsx

import React, { useEffect, useState } from 'react';
import { Card, Spin } from 'antd';
import * as d3 from 'd3';

interface CoverageData {
  grade_level: number;
  topic_area: string;
  question_count: number;
}

const CurriculumCoverage: React.FC = () => {
  const [data, setData] = useState<CoverageData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchCoverageData();
  }, []);

  const fetchCoverageData = async () => {
    try {
      const response = await fetch('/api/curriculum-coverage');
      const coverageData = await response.json();
      setData(coverageData);
      renderHeatmap(coverageData);
    } catch (error) {
      console.error('Failed to fetch coverage data:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderHeatmap = (data: CoverageData[]) => {
    // D3.js heatmap implementation
    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Clear existing SVG
    d3.select('#coverage-heatmap').selectAll('*').remove();

    const svg = d3.select('#coverage-heatmap')
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Define scales
    const grades = [1, 2, 3, 4, 5, 6];
    const topics = Array.from(new Set(data.map(d => d.topic_area)));

    const xScale = d3.scaleBand()
      .domain(grades.map(String))
      .range([0, width])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(topics)
      .range([0, height])
      .padding(0.05);

    const colorScale = d3.scaleSequential()
      .domain([0, d3.max(data, d => d.question_count) || 20])
      .interpolator(d3.interpolateBlues);

    // Draw rectangles
    svg.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.grade_level)) || 0)
      .attr('y', d => yScale(d.topic_area) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.question_count))
      .attr('stroke', 'white')
      .attr('stroke-width', 2)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('opacity', 0.7);
        // Show tooltip
      })
      .on('mouseout', function() {
        d3.select(this).attr('opacity', 1);
      });

    // Add text labels
    svg.selectAll('text.cell-label')
      .data(data)
      .enter()
      .append('text')
      .attr('class', 'cell-label')
      .attr('x', d => (xScale(String(d.grade_level)) || 0) + xScale.bandwidth() / 2)
      .attr('y', d => (yScale(d.topic_area) || 0) + yScale.bandwidth() / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', d => d.question_count > 10 ? 'white' : 'black')
      .attr('font-size', '12px')
      .attr('font-weight', 'bold')
      .text(d => d.question_count);

    // Add axes
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .style('font-size', '14px');

    svg.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('font-size', '12px');

    // Add axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height + 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Grade Level');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', -60)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .text('Topic Area');
  };

  return (
    <Card title="Curriculum Coverage Heatmap" style={{ marginTop: 20 }}>
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px' }}>
          <Spin size="large" />
        </div>
      ) : (
        <div id="coverage-heatmap" />
      )}
    </Card>
  );
};

export default CurriculumCoverage;
