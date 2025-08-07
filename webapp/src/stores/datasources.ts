import { defineStore } from 'pinia';
import { getApi, getApiUi, postApiUi } from 'src/boot/axios';
import { ref } from 'vue';

export enum DataSourceType {
  /** Data provided in external resource (Google Spreadsheet) */
  External = 'External',
  /** Data uploaded as is (CSV, JSON, entered inline) */
  Internal = 'Internal',
}

/**
 * Column configuration.
 */
export interface ColumnSchema {
  geoColumn: string;
  dateColumn: string;
  metricColumns: string[];
  costColumn?: string;
}

/**
 * Combined DataSource with optional nested data.
 */
export interface DataSource {
  id: string;
  name: string;
  description?: string;
  sourceType: DataSourceType;
  columns: ColumnSchema;
  sourceLink: string;
  createdAt?: Date;
  updatedAt?: Date;

  // Data - optional because it might not be loaded yet for external sources
  data?: DataSourceData;
}

export interface DataSourceData {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rawRows: any[];
  // Pre-processed views
  geoUnits?: string[];
  uniqueDates?: string[];
  metricNames?: string[];
  metrics?: Array<{
    name: string;
    min: number;
    max: number;
  }>;
  costSum?: number; // Sum of the cost column, if available
}

export type GetDataSourcesResponse = DataSource[];

// Transform functions to handle API communication
function transformToApi(source: DataSource) {
  return {
    id: source.id,
    name: source.name,
    description: source.description,
    source_type: source.sourceType,
    columns: {
      geo_column: source.columns.geoColumn,
      date_column: source.columns.dateColumn,
      metric_columns: source.columns.metricColumns,
      cost_column: source.columns.costColumn,
    },
    source_link: source.sourceLink,
    created_at: source.createdAt,
    updated_at: source.updatedAt,
    data: source.data?.rawRows,
  };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformFromApi(apiData: any): DataSource {
  return {
    id: apiData.id,
    name: apiData.name,
    description: apiData.description,
    sourceType: apiData.source_type,
    columns: {
      geoColumn: apiData.columns.geo_column,
      dateColumn: apiData.columns.date_column,
      metricColumns: apiData.columns.metric_columns,
      costColumn: apiData.columns.cost_column,
    },
    sourceLink: apiData.source_link,
    createdAt: apiData.created_at ? new Date(apiData.created_at) : undefined,
    updatedAt: apiData.updated_at ? new Date(apiData.updated_at) : undefined,
    data: apiData.data ? { rawRows: apiData.data } : undefined,
  };
}

export const useDataSourcesStore = defineStore('datasources', () => {
  const datasources = ref<DataSource[]>([]);
  const loading = ref(false);
  const error = ref<string | null>(null);
  const isLoaded = ref(false);

  // Get a data source by ID from local state
  function getDataSourceById(id: string) {
    return datasources.value.find((d) => d.id === id) || null;
  }

  // Create a new empty data source template
  function createEmptyDataSource(): DataSource {
    return {
      id: '', // Will be assigned by the server
      name: '',
      description: '',
      sourceType: DataSourceType.Internal,
      columns: {
        geoColumn: '',
        dateColumn: '',
        metricColumns: [],
        costColumn: undefined,
      },
      sourceLink: '',
      createdAt: undefined,
      updatedAt: undefined,
    };
  }

  /**
   * Save a data source (add or update) in local state and on the server.
   * @param dataSource a datasource to save (either existing or new)
   */
  async function saveDataSource(dataSource: DataSource) {
    const existingIndex = datasources.value.findIndex((d) => d.id === dataSource.id);
    const now = new Date();
    dataSource.updatedAt = now;
    if (!dataSource.createdAt) {
      dataSource.createdAt = now;
    }
    loading.value = true;
    error.value = null;

    try {
      // Transform to API format
      const apiDataSource = transformToApi(dataSource);

      let response;
      if (existingIndex >= 0) {
        // Update existing data source
        response = await postApiUi(
          `datasources/${dataSource.id}`,
          apiDataSource,
          'Updating data source',
          { method: 'PUT' },
        );
        if (!response) return undefined;

        // Transform from API format
        const updatedDataSource = response.data ? transformFromApi(response.data) : dataSource;
        updatedDataSource.data = normalizeRawData(
          updatedDataSource.data.rawRows,
          updatedDataSource.columns,
        );
        // Preserve data if not included in response
        if (!updatedDataSource.data && datasources.value[existingIndex].data) {
          updatedDataSource.data = datasources.value[existingIndex].data;
        }

        datasources.value[existingIndex] = updatedDataSource;
        return updatedDataSource;
      } else {
        // Creating a new data source
        response = await postApiUi('datasources', apiDataSource, 'Saving data source');
        if (!response) return undefined;

        // If the server returned a new version, use it, otherwise the one we sent
        const newDataSource = response.data ? transformFromApi(response.data) : dataSource;
        newDataSource.data = normalizeRawData(newDataSource.data.rawRows, newDataSource.columns);
        datasources.value.push(newDataSource);
        return newDataSource;
      }
    } catch (err) {
      console.error('Failed to save data source:', err);
      error.value = 'Failed to save data source';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  /**
   * Load all data sources from the server.
   */
  async function loadDataSources(interactive = true, reload = false) {
    if (isLoaded.value && !reload) {
      return datasources.value;
    }

    loading.value = true;
    error.value = null;

    try {
      const response = await (interactive
        ? getApiUi<GetDataSourcesResponse>('datasources', undefined, 'Fetching data sources')
        : getApi<GetDataSourcesResponse>('datasources'));

      if (!response) return undefined;

      datasources.value = response.data.map(transformFromApi);
      isLoaded.value = true;
      return datasources.value;
    } catch (err) {
      console.error('Failed to load data sources from server:', err);
      error.value = err;
      throw err;
    } finally {
      loading.value = false;
    }
  }

  /**
   * Delete a data source (both locally and on the server)
   * @param id data source id
   */
  async function deleteDataSource(id: string) {
    loading.value = true;
    error.value = null;

    try {
      // delete from server
      const response = await postApiUi(`datasources/${id}`, undefined, 'Deleting data source', {
        method: 'DELETE',
      });
      if (!response) return false;

      // remove locally
      datasources.value = datasources.value.filter((d) => d.id !== id);

      return true;
    } catch (err) {
      console.error('Failed to delete data source:', err);
      error.value = 'Failed to delete data source';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  // Load data for an external data source
  async function loadDataSourceData(
    dataSource: DataSource,
    forceReload = false,
  ): Promise<DataSourceData> {
    error.value = null;
    if (dataSource.data && !forceReload) return dataSource.data;

    loading.value = true;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const response = await getApiUi<any[]>(
        `datasources/${dataSource.id}/data`,
        undefined,
        'Loading data source data',
      );

      if (!response) return;
      dataSource.data = normalizeRawData(response.data, dataSource.columns);

      return dataSource.data;
    } catch (err) {
      console.error('Failed to load data source data:', err);
      error.value = 'Failed to load data source data';
      throw err;
    } finally {
      loading.value = false;
    }
  }

  // Calculate statistics from raw rows using column schema
  function normalizeRawData(rawData: unknown[], columns: ColumnSchema): DataSourceData {
    if (!rawData || rawData.length === 0) {
      return {
        rawRows: [],
        geoUnits: [],
        uniqueDates: [],
        metricNames: [],
        metrics: [],
        costSum: 0,
      };
    }

    const { geoColumn, dateColumn, metricColumns, costColumn } = columns;

    // Extract unique geo units and dates
    const geoUnits = [...new Set(rawData.map((row) => row[geoColumn]))];
    const uniqueDates = [...new Set(rawData.map((row) => row[dateColumn]))].sort();

    // Calculate statistics for each metric
    const metrics = metricColumns.map((metricName) => {
      const values = rawData
        .map((row) => row[metricName])
        .filter((val): val is number => typeof val === 'number');

      if (values.length === 0) return { name: metricName, min: 0, max: 0 };

      return {
        name: metricName,
        min: Math.min(...values),
        max: Math.max(...values),
      };
    });

    let costSum = 0;
    if (costColumn) {
      costSum = rawData.reduce((sum: number, row) => {
        const costVal = Number(row[costColumn]);
        return sum + (isNaN(costVal) ? 0 : costVal);
      }, 0) as number;
    }

    return {
      rawRows: rawData,
      geoUnits,
      uniqueDates,
      metricNames: metricColumns,
      metrics,
      costSum,
    };
  }

  return {
    datasources,
    loading,
    error,
    isLoaded,

    getDataSourceById,
    createEmptyDataSource,
    saveDataSource,
    loadDataSources,
    deleteDataSource,
    loadDataSourceData,
    normalizeRawData,
  };
});
