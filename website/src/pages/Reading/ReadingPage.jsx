import { useState, useEffect, useMemo } from 'react';
import { Book, Book as BookIcon, List, Grid, Clock, Calendar, Tag, User, Star } from 'lucide-react';
import { useData } from '../../context/DataContext';
import { usePageTitle } from '../../hooks/usePageTitle';

// Import components
import BookDetails from './components/BookDetails';
import BookCard from './components/BookCard';

// Import standardized UI components
import {
  FilteringPanel,
  Filter,
  PageWrapper,
  PageHeader,
  TabNavigation,
  ContentTab,
  AnalysisTab,
  KPICardsPanel,
  ContentCardsGroup
} from '../../components/ui';
import KpiCard from '../../components/charts/KpiCard';

// Import chart components for analysis tab
import TimeSeriesBarChart from '../../components/charts/TimeSeriesBarChart';
import IntensityHeatmap from '../../components/charts/IntensityHeatmap';
import TopChart from '../../components/charts/TopChart';
import ProportionChart from '../../components/charts/ProportionChart';

const ReadingPage = () => {
  usePageTitle('Reading');
  const { data, loading, error, fetchData } = useData();
  const [books, setBooks] = useState([]);
  const [filteredBooks, setFilteredBooks] = useState([]);
  const [readingEntries, setReadingEntries] = useState([]);
  const [filteredReadingEntries, setFilteredReadingEntries] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);

  const [viewMode, setViewMode] = useState('grid');
  const [selectedBook, setSelectedBook] = useState(null);
  const [activeTab, setActiveTab] = useState('content');

  // Fetch reading data when component mounts
  useEffect(() => {
    if (typeof fetchData === 'function') {
      Promise.all([
        fetchData('readingBooks'),
        fetchData('readingSessions')
      ]);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Process books data when it's loaded from new dual-file structure
  useEffect(() => {
    if (data?.readingBooks && data?.readingSessions) {
      setIsProcessing(true);

      // Convert timestamp strings to Date objects for JavaScript date operations
      const processedBooks = data.readingBooks.map(book => ({
        ...book,
        timestamp: book.timestamp ? new Date(book.timestamp) : null
      }));

      const processedSessions = data.readingSessions.map(session => ({
        ...session,
        timestamp: session.timestamp ? new Date(session.timestamp) : null
      }));

      // ContentTab now handles sorting internally
      setBooks(processedBooks);
      setFilteredBooks(processedBooks);
      setReadingEntries(processedSessions);
      setFilteredReadingEntries(processedSessions);
      setIsProcessing(false);
    }
  }, [data?.readingBooks, data?.readingSessions]);

  // Apply filters when FilteringPanel filters change
  // FilteringPanel now returns pre-filtered data per source!
  const handleFiltersChange = (filteredDataSources) => {
    // ContentTab now handles sorting internally
    setFilteredBooks(filteredDataSources.readingBooks || []);
    setFilteredReadingEntries(filteredDataSources.readingSessions || []);
  };

  const handleBookClick = (book) => {
    setSelectedBook(book);
  };

  const handleCloseDetails = () => {
    setSelectedBook(null);
  };

  // Memoize data object to prevent FilteringPanel re-renders
  const filterPanelData = useMemo(() => ({
    readingBooks: books,
    readingSessions: readingEntries
  }), [books, readingEntries]);

  return (
    <PageWrapper
      error={error?.readingBooks || error?.readingSessions}
      errorTitle="Reading Tracker"
    >
      {/* Page Header */}
      <PageHeader
        title="Reading Tracker"
        description="Track your reading habits and discover insights about your books"
      />

        {!(loading?.readingBooks || loading?.readingSessions) && !isProcessing && (
          <>
            {/* FilteringPanel with Filter children */}
            <FilteringPanel
              data={filterPanelData}
              onFiltersChange={handleFiltersChange}
              renderMode="bubble"
              bubbleConfig={{
                maxVisible: 10,
                searchThreshold: 15,
                showCounts: false
              }}
            >
              <Filter
                type="daterange"
                label="Reading Date"
                field="timestamp"
                icon={<Calendar />}
                dataSources={['readingBooks', 'readingSessions']}
              />
              <Filter
                type="multiselect"
                label="Reading Year"
                field="reading_year"
                icon={<Clock />}
                placeholder="Select years"
                dataSources={['readingBooks', 'readingSessions']}
              />
              <Filter
                type="multiselect"
                label="Type"
                field="fiction_yn"
                icon={<Book />}
                dataSources={['readingBooks']}
                options={['Fiction', 'Non-Fiction']}
              />
              <Filter
                type="multiselect"
                label="Format"
                field="reading_format"
                icon={<Tag />}
                placeholder="Select how book was read"
                dataSources={['readingBooks', 'readingSessions']}
              />
              <Filter
                type="multiselect"
                label="Genre"
                field="genre"
                icon={<Tag />}
                placeholder="Select genres"
                dataSources={['readingBooks', 'readingSessions']}
              />
              <Filter
                type="multiselect"
                label="Author"
                field="author"
                icon={<User />}
                placeholder="Select authors"
                dataSources={['readingBooks']}
              />

              <Filter
                type="multiselect"
                label="Rating"
                field="my_rating"
                icon={<Star />}
                placeholder="Select ratings"
                dataSources={['readingBooks']}
              />
            </FilteringPanel>

            {/* Statistics Cards with KpiCard children */}
            <KPICardsPanel
              dataSources={{
                readingBooks: filteredBooks,
                readingSessions: filteredReadingEntries
              }}
              loading={loading?.readingBooks || loading?.readingSessions}
            >
              <KpiCard
                dataSource="readingBooks"
                metricOptions={{
                  label: 'Books Read',
                  aggregation: 'count'
                }}
                icon={<Book />}
              />
              <KpiCard
                dataSource="readingBooks"
                metricOptions={{
                  label: 'Total Pages',
                  aggregation: 'sum',
                  field: 'number_of_pages',
                  decimals: 0
                }}
                icon={<BookIcon />}
              />
              <KpiCard
                dataSource="readingBooks"
                metricOptions={{
                  label: 'Average Rating',
                  aggregation: 'average',
                  field: 'my_rating',
                  decimals: 1,
                  filterConditions: [{ field: 'my_rating', operator: '>', value: 0 }]
                }}
                icon={<Star />}
              />
              <KpiCard
                dataSource="readingBooks"
                metricOptions={{
                  label: 'Avg. Pages per day',
                  aggregation: 'average',
                  field: 'pages_per_day',
                  decimals: 0,
                  filterConditions: [{ field: 'pages_per_day', operator: '>', value: 0 }]
                }}
                icon={<Clock />}
              />
            </KPICardsPanel>

            {/* Tab Navigation */}
            <TabNavigation
              contentLabel="Books"
              contentIcon={BookIcon}
              activeTab={activeTab}
              onTabChange={setActiveTab}
            />
          </>
        )}

        {/* Books Tab Content */}
        {activeTab === 'content' && (
          <ContentTab
            loading={loading?.readingBooks || loading?.readingSessions || isProcessing}
            viewMode={viewMode}
            onViewModeChange={setViewMode}
            viewModes={[
              { mode: 'grid', icon: Grid },
              { mode: 'list', icon: List }
            ]}
            items={filteredBooks}
            loadingIcon={Book}
            emptyState={{
              icon: BookIcon,
              title: "No books found",
              message: "No books match your current filters. Try adjusting your criteria."
            }}
            sortOptions={[
              { value: 'timestamp', label: 'Finish Date', type: 'date' },
              { value: 'title', label: 'Title', type: 'string' },
              { value: 'author', label: 'Author', type: 'string' },
              { value: 'my_rating', label: 'Rating', type: 'number' },
              { value: 'number_of_pages', label: 'Pages', type: 'number' },
              { value: 'pages_per_day', label: 'Pages per Day', type: 'number' }
            ]}
            defaultSortField="timestamp"
            defaultSortDirection="desc"
            renderGrid={(books) => (
              <ContentCardsGroup
                items={books}
                viewMode="grid"
                renderItem={(book) => (
                  <BookCard
                    key={book.book_id}
                    book={book}
                    viewMode="grid"
                    onClick={handleBookClick}
                  />
                )}
              />
            )}
            renderList={(books) => (
              <ContentCardsGroup
                items={books}
                viewMode="list"
                renderItem={(book) => (
                  <BookCard
                    key={`list-${book.book_id}`}
                    book={book}
                    viewMode="list"
                    onClick={handleBookClick}
                  />
                )}
              />
            )}
          />
        )}

        {/* Analysis Tab Content */}
        {activeTab === 'analysis' && (
          <AnalysisTab
            renderCharts={() => (
              <>
                <TimeSeriesBarChart
                  data={filteredReadingEntries}
                  dateColumnName="timestamp"
                  metricOptions={[
                    { value: 'books', label: 'Books', aggregation: 'count_distinct', field: 'book_id', decimals: 0 },
                    { value: 'pages', label: 'Pages', aggregation: 'sum', field: 'page_split', decimals: 0 },
                    { value: 'pages per session', label: 'Avg Pages per Session', aggregation: 'average', field: 'page_split', decimals: 0 },
                    { value: 'rating', label: 'Avg Rating', aggregation: 'average', field: 'my_rating', suffix: '★', decimals: 1 }
                  ]}
                  defaultMetric="pages"
                  title="Reading Activity by Period"
                />
                <IntensityHeatmap
                  data={filteredReadingEntries}
                  dateColumnName="timestamp"
                  valueColumnName="page_split"
                  title="Reading Activity by Day and Time"
                  treatMidnightAsUnknown={true}
                />
                <TopChart
                  data={filteredBooks}
                  dimensionOptions={[
                    { value: 'author', label: 'Author', field: 'author', labelFields: ['author'] },
                    { value: 'genre', label: 'Genre', field: 'genre', labelFields: ['genre'] },
                    { value: 'title', label: 'Book', field: 'title', labelFields: ['title'] },
                    { value: 'format', label: 'Format', field: 'reading_format', labelFields: ['reading_format'] }
                  ]}
                  metricOptions={[
                    { value: 'pages', label: 'Total Pages', aggregation: 'sum', field: 'number_of_pages', suffix: ' pages', decimals: 0 },
                    { value: 'books', label: 'Total Books', aggregation: 'count_distinct', field: 'book_id', suffix: ' books', decimals: 0 },
                    { value: 'pages per day', label: 'Avg pages per day', aggregation: 'average', field: 'pages_per_day', suffix: ' pages per day', decimals: 0 },
                    { value: 'reading_duration_final', label: 'Completion Time', aggregation: 'sum', field: 'reading_duration_final', suffix: ' days', decimals: 0 },
                    { value: 'my_rating', label: 'Avg Rating', aggregation: 'average', field: 'my_rating', suffix: '★', decimals: 1 }
                  ]}
                  defaultDimension="author"
                  defaultMetric="pages"
                  title="Top Books Analysis"
                  topN={10}
                  imageField="cover_url"
                  enableTopNControl={true}
                  topNOptions={[5, 10, 15, 20, 25, 30]}
                  enableSortToggle={true}
                  scrollable={true}
                  barHeight={50}
                />
                <ProportionChart
                  data={filteredBooks}
                  dimensionOptions={[
                    { value: 'genre', label: 'Genre', field: 'genre' },
                    { value: 'author', label: 'Author', field: 'author' },
                    { value: 'reading_format', label: 'Format', field: 'reading_format' },
                    { value: 'fiction_yn', label: 'Fiction/Non-Fiction', field: 'fiction_yn' }
                  ]}
                  metricOptions={[
                    { value: 'pages', label: 'Pages', aggregation: 'sum', field: 'number_of_pages', suffix: ' pages', decimals: 0 },
                    { value: 'books', label: 'Books', aggregation: 'count', field: 'book_id' }
                  ]}
                  defaultDimension="genre"
                  defaultMetric="pages"
                  title="Reading Distribution"
                  maxCategories={8}
                  showPercentages={true}
                />
              </>
            )}
          />
        )}

      {/* Book Details Modal */}
      {selectedBook && (
        <BookDetails book={selectedBook} onClose={handleCloseDetails} />
      )}
    </PageWrapper>
  );
};

export default ReadingPage;
