import { defineStore } from 'pinia';
import { ref } from 'vue';

export enum DataSourceType {
  /** Data provided in external resource (Google Spreadsheet) */
  External = 'External',
  /** Data uploaded as is (CSV, JSON, entered inline) */
  Internal = 'Internal',
}

export interface DataRow {
  geoUnit: string;
  date: string;
  metric: number;
  cost?: number;
}

/**
 * Combined DataSource with optional nested data.
 */
export interface DataSource {
  id: string;
  name: string;
  description?: string;
  sourceType: DataSourceType;
  geoColumn: string;
  dateColumn: string;
  metricColumns: string[];
  costColumn?: string;

  // For external sources - only need URL for Google Sheets or similar
  externalSourceUrl?: string;

  // Metadata
  createdAt?: Date;
  updatedAt?: Date;

  // Data - optional because it might not be loaded yet for external sources
  data?: {
    rows: Record<string, DataRow[]>;
    geo: string[];
    numberOfDays: number;
  };
}

export const useDataSourcesStore = defineStore('datasources', () => {
  const datasources = ref<DataSource[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);

  // Track server sync status
  const isSynced = ref(true);

  // LOCAL OPERATIONS - manipulate state in memory

  // Add a new data source to local state
  const addDataSource = (dataSource: DataSource) => {
    // Update timestamps
    const now = new Date();
    dataSource.updatedAt = now;

    if (!dataSource.createdAt) {
      dataSource.createdAt = now;
    }

    datasources.value.push({ ...dataSource });
    isSynced.value = false;
    return dataSource;
  };

  // Update an existing data source in local state
  const updateDataSource = (dataSource: DataSource) => {
    const existingIndex = datasources.value.findIndex((d) => d.id === dataSource.id);

    if (existingIndex === -1) {
      throw new Error(`Data source with ID ${dataSource.id} not found`);
    }

    // Update timestamp
    dataSource.updatedAt = new Date();

    // Preserve data if not provided
    if (!dataSource.data && datasources.value[existingIndex].data) {
      dataSource.data = datasources.value[existingIndex].data;
    }

    datasources.value[existingIndex] = { ...dataSource };
    isSynced.value = false;
    return dataSource;
  };

  // Remove a data source from local state
  const removeDataSource = (id: string) => {
    datasources.value = datasources.value.filter((d) => d.id !== id);
    isSynced.value = false;
    return true;
  };

  // Get a data source by ID from local state
  const getDataSourceById = (id: string) => {
    return datasources.value.find((d) => d.id === id) || null;
  };

  // Save a data source (add or update) in local state
  const saveDataSource = (dataSource: DataSource) => {
    const existingIndex = datasources.value.findIndex((d) => d.id === dataSource.id);

    if (existingIndex >= 0) {
      return updateDataSource(dataSource);
    } else {
      return addDataSource(dataSource);
    }
  };

  // Load data for an external data source
  const loadDataSourceData = async (id: string) => {
    loading.value = true;
    error.value = null;

    try {
      const dataSource = getDataSourceById(id);

      if (!dataSource) {
        throw new Error(`Data source with ID ${id} not found`);
      }

      if (dataSource.sourceType === DataSourceType.External) {
        // In a real app, you would fetch data from the external source based on the URL
        console.log(`Fetching data from: ${dataSource.externalSourceUrl}`);

        // For now, we'll just simulate a delay and provide sample data
        await new Promise((resolve) => setTimeout(resolve, 1000));

        // Simulate fetching data from the external source
        const externalData = {
          rows: {
            Germany: [
              {
                geoUnit: 'Germany',
                date: '2023-03-01',
                metric: 2500,
                cost: 1200,
              },
              {
                geoUnit: 'Germany',
                date: '2023-03-08',
                metric: 2600,
                cost: 1250,
              },
              {
                geoUnit: 'Germany',
                date: '2023-03-15',
                metric: 2700,
                cost: 1300,
              },
            ],
            France: [
              {
                geoUnit: 'France',
                date: '2023-03-01',
                metric: 1800,
                cost: 900,
              },
              {
                geoUnit: 'France',
                date: '2023-03-08',
                metric: 1900,
                cost: 950,
              },
              {
                geoUnit: 'France',
                date: '2023-03-15',
                metric: 2000,
                cost: 1000,
              },
            ],
            UK: [
              { geoUnit: 'UK', date: '2023-03-01', metric: 2200, cost: 1100 },
              { geoUnit: 'UK', date: '2023-03-08', metric: 2300, cost: 1150 },
              { geoUnit: 'UK', date: '2023-03-15', metric: 2400, cost: 1200 },
            ],
          },
          geo: ['Germany', 'France', 'UK'],
          numberOfDays: 3,
        };

        // Update the data source with the fetched data
        dataSource.data = externalData;

        // Update in the store
        updateDataSource(dataSource);
      }

      return dataSource;
    } catch (err) {
      console.error('Failed to load data source data:', err);
      error.value = 'Failed to load data source data';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  // SERVER OPERATIONS - interact with backend

  // Load all data sources from the server
  const loadDataSources = async () => {
    loading.value = true;
    error.value = null;

    try {
      // In a real app, you would fetch data from your API
      // For example:
      // const response = await fetch('/api/datasources');
      // const data = await response.json();
      // datasources.value = data;

      // For demonstration, we'll use some sample data
      if (datasources.value.length === 0) {
        // Only load sample data if there are no data sources yet
        await new Promise((resolve) => setTimeout(resolve, 500));

        datasources.value = [
          {
            id: '1',
            name: 'US States Campaign',
            description: 'Marketing campaign data across US states',
            sourceType: DataSourceType.Internal,
            geoColumn: 'State',
            dateColumn: 'Date',
            metricColumns: ['Visits', 'Conversions'],
            costColumn: 'Cost',
            createdAt: new Date('2023-01-15'),
            updatedAt: new Date('2023-02-10'),
            data: {
              rows: {
                CA: [
                  {
                    geoUnit: 'CA',
                    date: '2023-01-01',
                    metric: 1200,
                    cost: 500,
                  },
                  {
                    geoUnit: 'CA',
                    date: '2023-01-02',
                    metric: 1300,
                    cost: 550,
                  },
                  {
                    geoUnit: 'CA',
                    date: '2023-01-03',
                    metric: 1250,
                    cost: 525,
                  },
                ],
                NY: [
                  { geoUnit: 'NY', date: '2023-01-01', metric: 950, cost: 400 },
                  {
                    geoUnit: 'NY',
                    date: '2023-01-02',
                    metric: 1000,
                    cost: 425,
                  },
                  {
                    geoUnit: 'NY',
                    date: '2023-01-03',
                    metric: 1050,
                    cost: 450,
                  },
                ],
              },
              geo: ['CA', 'NY'],
              numberOfDays: 3,
            },
          },
          {
            id: '2',
            name: 'EU Countries Data',
            description: 'Performance metrics for European countries',
            sourceType: DataSourceType.External,
            geoColumn: 'Country',
            dateColumn: 'Week',
            metricColumns: ['Impressions', 'Clicks', 'Purchases'],
            externalSourceUrl: 'https://docs.google.com/spreadsheets/d/example',
            createdAt: new Date('2023-03-20'),
            updatedAt: new Date('2023-03-20'),
            // No data loaded yet for this external source
          },
        ];
      }

      isSynced.value = true;
      return datasources.value;
    } catch (err) {
      console.error('Failed to load data sources from server:', err);
      error.value = 'Failed to load data sources from server';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  // Save all data sources to the server
  const saveToServer = async () => {
    if (isSynced.value) {
      console.log('Already in sync with server, no need to save');
      return;
    }

    loading.value = true;
    error.value = null;

    try {
      // In a real app, you would save all data sources to your API
      // For example:
      // await fetch('/api/datasources', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(datasources.value)
      // });

      // Simulate a delay for the API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      console.log('Saved all data sources to server:', datasources.value);
      isSynced.value = true;

      return true;
    } catch (err) {
      console.error('Failed to save data sources to server:', err);
      error.value = 'Failed to save data sources to server';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  // Delete a data source (both locally and on the server)
  const deleteDataSource = async (id: string) => {
    loading.value = true;
    error.value = null;

    try {
      // First remove locally
      removeDataSource(id);

      // Then delete from server
      // In a real app:
      // await fetch(`/api/datasources/${id}`, { method: 'DELETE' });

      // Simulate server call
      await new Promise((resolve) => setTimeout(resolve, 300));

      isSynced.value = true;
      return true;
    } catch (err) {
      console.error('Failed to delete data source:', err);
      error.value = 'Failed to delete data source';
      throw err;
    } finally {
      loading.value = false;
    }
  };

  return {
    datasources,
    loading,
    error,
    isSynced,

    // Local operations
    addDataSource,
    updateDataSource,
    removeDataSource,
    getDataSourceById,
    saveDataSource,
    loadDataSourceData,

    // Server operations
    loadDataSources,
    saveToServer,
    deleteDataSource,
  };
});
