
import React, { useState, useEffect } from 'react';
import { Table, Button, Tag, Space, Input, Select, Pagination } from 'antd';
import { SearchOutlined, FilterOutlined, DownloadOutlined } from '@ant-design/icons';




interface Question {
  id: string;
  question_stem: string;
  bloom_level: string;
  estimated_difficulty: number;
  topic_area: string;
  grade_level: number;
  validation_status: string;
  overall_quality?: number;
}

const QuestionBrowser: React.FC = () => {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState({
    grade_level: null,
    topic_area: null,
    bloom_level: null,
    difficulty_min: 1,
    difficulty_max: 5,
    search_text: ''
  });
  const [pagination, setPagination] = useState({
    current: 1,
    pageSize: 20,
    total: 0
  });

  // Fetch questions from API
  useEffect(() => {
    fetchQuestions();
  }, [filters, pagination.current]);

  const fetchQuestions = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...filters,
          page: pagination.current,
          page_size: pagination.pageSize
        })
      });
      const data = await response.json();
      setQuestions(data.questions);
      setPagination({ ...pagination, total: data.total });
    } catch (error) {
      console.error('Failed to fetch questions:', error);
    } finally {
      setLoading(false);
    }
  };

  // Table columns configuration
  const columns = [
    {
      title: 'Question',
      dataIndex: 'question_stem',
      key: 'question_stem',
      width: '40%',
      ellipsis: true,
      render: (text: string, record: Question) => (
        <a onClick={() => openQuestionDetail(record.id)}>{text}</a>
      )
    },
    {
      title: 'Grade',
      dataIndex: 'grade_level',
      key: 'grade_level',
      width: '8%',
      sorter: true,
      render: (grade: number) => <Tag color="blue">Grade {grade}</Tag>
    },
    {
      title: 'Topic',
      dataIndex: 'topic_area',
      key: 'topic_area',
      width: '15%',
      filters: [
        { text: 'Online Safety', value: 'Online Safety' },
        { text: 'Password Security', value: 'Password Security' },
        { text: 'Phishing Recognition', value: 'Phishing Recognition' },
        { text: 'Digital Footprint', value: 'Digital Footprint' },
        { text: 'Digital Citizenship', value: 'Digital Citizenship' },
        { text: 'Cybersecurity Basics', value: 'Cybersecurity Basics' }
      ],
      onFilter: (value, record) => record.topic_area === value
    },
    {
      title: 'Bloom\'s Level',
      dataIndex: 'bloom_level',
      key: 'bloom_level',
      width: '12%',
      render: (level: string) => {
        const colors = {
          'Knowledge': 'green',
          'Comprehension': 'cyan',
          'Application': 'orange',
          'Analysis': 'red'
        };
        return <Tag color={colors[level]}>{level}</Tag>;
      }
    },
    {
      title: 'Difficulty',
      dataIndex: 'estimated_difficulty',
      key: 'estimated_difficulty',
      width: '10%',
      sorter: true,
      render: (diff: number) => (
        <span>{diff.toFixed(1)}/5.0</span>
      )
    },
    {
      title: 'Quality',
      dataIndex: 'overall_quality',
      key: 'overall_quality',
      width: '10%',
      render: (quality: number | undefined) => (
        quality ? (
          <Tag color={quality >= 4.0 ? 'green' : quality >= 3.0 ? 'orange' : 'red'}>
            {quality.toFixed(1)}/5.0
          </Tag>
        ) : <Tag color="gray">Not Rated</Tag>
      )
    },
    {
      title: 'Actions',
      key: 'actions',
      width: '15%',
      render: (_, record: Question) => (
        <Space size="small">
          <Button size="small" onClick={() => openQuestionDetail(record.id)}>
            View
          </Button>
          <Button size="small" onClick={() => openQuestionEditor(record.id)}>
            Edit
          </Button>
          <Button size="small" danger onClick={() => deleteQuestion(record.id)}>
            Delete
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div className="question-browser">
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* Filter Controls */}
        <Space wrap>
          <Select
            placeholder="Grade Level"
            style={{ width: 120 }}
            allowClear
            onChange={(value) => setFilters({ ...filters, grade_level: value })}
          >
            {[1, 2, 3, 4, 5, 6].map(grade => (
              <Select.Option key={grade} value={grade}>Grade {grade}</Select.Option>
            ))}
          </Select>

          <Select
            placeholder="Topic Area"
            style={{ width: 180 }}
            allowClear
            onChange={(value) => setFilters({ ...filters, topic_area: value })}
          >
            <Select.Option value="Online Safety">Online Safety</Select.Option>
            <Select.Option value="Password Security">Password Security</Select.Option>
            <Select.Option value="Phishing Recognition">Phishing Recognition</Select.Option>
            <Select.Option value="Digital Footprint">Digital Footprint</Select.Option>
            <Select.Option value="Digital Citizenship">Digital Citizenship</Select.Option>
            <Select.Option value="Cybersecurity Basics">Cybersecurity Basics</Select.Option>
          </Select>

          <Select
            placeholder="Bloom's Level"
            style={{ width: 150 }}
            allowClear
            onChange={(value) => setFilters({ ...filters, bloom_level: value })}
          >
            <Select.Option value="Knowledge">Knowledge</Select.Option>
            <Select.Option value="Comprehension">Comprehension</Select.Option>
            <Select.Option value="Application">Application</Select.Option>
            <Select.Option value="Analysis">Analysis</Select.Option>
          </Select>

          <Input.Search
            placeholder="Search questions..."
            style={{ width: 250 }}
            onSearch={(value) => setFilters({ ...filters, search_text: value })}
            prefix={<SearchOutlined />}
          />

          <Button
            type="primary"
            icon={<DownloadOutlined />}
            onClick={exportSelected}
          >
            Export Selected
          </Button>
        </Space>

        {/* Question Table */}
        <Table
          columns={columns}
          dataSource={questions}
          loading={loading}
          rowKey="id"
          pagination={{
            ...pagination,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} questions`
          }}
          onChange={(newPagination, filters, sorter) => {
            setPagination({ ...pagination, current: newPagination.current });
          }}
        />
      </Space>
    </div>
  );
};

export default QuestionBrowser;
