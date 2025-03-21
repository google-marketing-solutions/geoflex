<template>
  <div class="q-pa-md">
    <q-stepper v-model="step" vertical color="primary" animated>
      <!-- Step 1: Basic Information -->
      <q-step :name="1" title="Basic Information" icon="settings" :done="step > 1">
        <q-card flat bordered>
          <q-card-section>
            <div class="row q-col-gutter-md">
              <div class="col-12 col-md-6">
                <q-input
                  v-model="currentDataSource.name"
                  label="Data Source Name *"
                  filled
                  :rules="[(val) => !!val || 'Name is required']"
                />
              </div>

              <div class="col-12 col-md-6" v-if="!isEditing">
                <q-select
                  v-model="currentDataSource.sourceType"
                  :options="sourceTypeOptions"
                  label="Data Source Type *"
                  filled
                  emit-value
                  map-options
                />
              </div>

              <div class="col-12">
                <q-input
                  v-model="currentDataSource.description"
                  label="Description"
                  type="textarea"
                  filled
                  autogrow
                />
              </div>
            </div>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Continue"
            @click="step = isEditing ? 3 : 2"
            :disable="!currentDataSource.name"
          />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 2: Data Import (only for new data sources) -->
      <q-step v-if="!isEditing" :name="2" title="Data Import" icon="cloud_upload" :done="step > 2">
        <q-card flat bordered>
          <q-card-section>
            <div v-if="currentDataSource.sourceType === DataSourceType.Internal">
              <p class="text-body1 q-mb-md">
                Upload a CSV file containing your geo-attributed time series data.
              </p>

              <q-file
                v-model="file"
                label="Select CSV File"
                filled
                counter
                accept=".csv"
                @update:model-value="handleFileUpload"
                :loading="loading"
              >
                <template v-slot:prepend>
                  <q-icon name="cloud_upload" />
                </template>
                <template v-slot:append v-if="file">
                  <q-icon name="close" @click.stop="clearFile" class="cursor-pointer" />
                </template>
                <template v-slot:hint>
                  CSV files with geo units, date units, and conversion metrics
                </template>
              </q-file>
            </div>

            <div v-else>
              <p class="text-body1 q-mb-md">Fetch data from an external data source.</p>

              <div class="row q-col-gutter-md">
                <div class="col-12">
                  <q-input
                    v-model="externalSourceUrl"
                    label="External Source URL (Google Sheets or similar) *"
                    filled
                    :rules="[(val) => !!val || 'URL is required']"
                  >
                    <template v-slot:prepend>
                      <q-icon name="link" />
                    </template>
                  </q-input>
                </div>
              </div>

              <q-btn
                color="primary"
                label="Fetch data"
                class="q-mt-md"
                @click="fetchExternalData"
                :disable="!externalSourceUrl"
                :loading="loadingExternalData"
              />
            </div>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Continue"
            @click="step = 3"
            :disable="!canContinueFromImport"
          />
          <q-btn flat color="primary" label="Back" class="q-mr-sm" @click="step = 1" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 3: Data Preview and Column Configuration -->
      <q-step :name="3" title="Data Preview and Column Configuration" icon="tune" :done="step > 3">
        <q-card flat bordered>
          <q-card-section>
            <div v-if="rawData.length && columns.length">
              <div class="row q-col-gutter-md">
                <!-- Column Configuration -->
                <div class="col-12 col-md-4">
                  <div class="text-h6 q-mb-md">Column Configuration</div>
                  <q-list bordered separator>
                    <q-item v-for="col in columns" :key="col.name">
                      <q-item-section>
                        <q-item-label>{{ col.name }}</q-item-label>
                      </q-item-section>
                      <q-item-section side>
                        <q-select
                          v-model="columnRoles[col.name]"
                          :options="columnRoleOptions"
                          dense
                          outlined
                          @update:model-value="handleColumnRoleChange(col.name, $event)"
                        />
                      </q-item-section>
                    </q-item>
                  </q-list>

                  <div v-if="!isColumnConfigValid" class="text-negative text-caption q-mt-sm">
                    Please select at least one Geo, one Date, and one Metric column
                  </div>
                </div>

                <!-- Data Preview -->
                <div class="col-12 col-md-8">
                  <div class="text-h6 q-mb-md">Data Preview</div>
                  <q-table
                    :rows="rawData"
                    :columns="dataPreviewColumns"
                    row-key="__index"
                    dense
                    :pagination="{ rowsPerPage: 10 }"
                  >
                    <template v-slot:top>
                      <q-input v-model="search" debounce="300" placeholder="Search" dense filled>
                        <template v-slot:append>
                          <q-icon name="search" />
                        </template>
                      </q-input>
                    </template>
                  </q-table>
                </div>
              </div>

              <div class="row q-col-gutter-md q-mt-lg">
                <div class="col-12 col-md-4">
                  <q-card class="bg-primary text-white">
                    <q-card-section>
                      <div class="text-h6">Total Rows</div>
                      <div class="text-h4">{{ rawData.length }}</div>
                    </q-card-section>
                  </q-card>
                </div>

                <div class="col-12 col-md-4">
                  <q-card class="bg-secondary text-white">
                    <q-card-section>
                      <div class="text-h6">Geo Units</div>
                      <div class="text-h4">{{ uniqueGeoCount }}</div>
                    </q-card-section>
                  </q-card>
                </div>

                <div class="col-12 col-md-4">
                  <q-card class="bg-accent text-white">
                    <q-card-section>
                      <div class="text-h6">Time Points</div>
                      <div class="text-h4">{{ uniqueDateCount }}</div>
                    </q-card-section>
                  </q-card>
                </div>
              </div>

              <div v-if="dataQualityIssues.length > 0" class="q-mt-md">
                <q-banner class="bg-warning text-white">
                  <template v-slot:avatar>
                    <q-icon name="warning" />
                  </template>
                  <div class="text-subtitle1 q-mb-sm">Data Quality Issues</div>
                  <ul class="q-mt-none q-mb-none">
                    <li v-for="(issue, index) in dataQualityIssues" :key="index">
                      {{ issue }}
                    </li>
                  </ul>
                </q-banner>
              </div>
            </div>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Continue"
            @click="step = 4"
            :disable="!isColumnConfigValid"
          />
          <q-btn
            flat
            color="primary"
            label="Back"
            class="q-mr-sm"
            @click="step = isEditing ? 1 : 2"
          />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 4: Save -->
      <q-step :name="4" title="Save" icon="save">
        <q-card flat bordered>
          <q-card-section>
            <div class="text-h6 q-mb-md">Review and Save</div>
            <p class="text-body1">
              Please review your data source configuration before saving. The data source will be
              saved with the following settings:
            </p>
            <ul class="q-mt-md">
              <li>Name: {{ currentDataSource.name }}</li>
              <li>Geo Column: {{ currentDataSource.columns.geoColumn }}</li>
              <li>Date Column: {{ currentDataSource.columns.dateColumn }}</li>
              <li>Metric Columns: {{ currentDataSource.columns.metricColumns.join(', ') }}</li>
              <li v-if="currentDataSource.columns.costColumn">
                Cost Column: {{ currentDataSource.columns.costColumn }}
              </li>
            </ul>
          </q-card-section>
        </q-card>

        <q-stepper-navigation>
          <q-btn
            color="primary"
            label="Save Data Source"
            @click="saveDataSource"
            :loading="saving"
            :disable="!isFormValid"
          />
          <q-btn flat color="primary" label="Back" class="q-mr-sm" @click="step = 3" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>
    </q-stepper>

    <!-- Error Dialog -->
    <q-dialog v-model="errorDialog" persistent>
      <q-card>
        <q-card-section class="row items-center">
          <q-avatar icon="error" color="negative" text-color="white" />
          <span class="q-ml-sm">{{ errorMessage }}</span>
        </q-card-section>
        <q-card-actions align="right">
          <q-btn flat label="Dismiss" color="primary" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useQuasar } from 'quasar';
import { parse } from 'csv-parse/browser/esm';
import type { DataSource } from 'stores/datasources';
import { useDataSourcesStore, DataSourceType } from 'stores/datasources';
//import { v4 as uuidv4 } from 'uuid';
import { getApiUi } from 'src/boot/axios';
import { assertIsError } from 'src/helpers/utils';

interface Props {
  dataSource?: DataSource | null;
}

const props = withDefaults(defineProps<Props>(), {
  dataSource: null,
});

const emit = defineEmits<{
  (e: 'saved', dataSource: DataSource): void;
  (e: 'canceled'): void;
}>();

const $q = useQuasar();

const dataSourcesStore = useDataSourcesStore();

// Step management
const step = ref(1);
const isEditing = computed(() => !!props.dataSource);

// Form data
const currentDataSource = ref<DataSource>(dataSourcesStore.createEmptyDataSource());
//const isNew = props.dataSource == null;

// UI state
const file = ref<File | null>(null);
const loading = ref(false);
const saving = ref(false);
const errorDialog = ref(false);
const errorMessage = ref('');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const rawData = ref<any[]>([]);
const columns = ref<{ name: string; label: string; field: string }[]>([]);
const search = ref('');
const loadingExternalData = ref(false);
const externalSourceUrl = ref('');

// Select options
const sourceTypeOptions = [
  { label: 'Internal (Upload CSV)', value: DataSourceType.Internal },
  { label: 'External (Google Sheets, etc.)', value: DataSourceType.External },
];

// Add new refs and computed properties after the existing ones
const columnRoles = ref<Record<string, ColumnRole>>({});

enum ColumnRole {
  Geo = 'Geo',
  Date = 'Date',
  Metric = 'Metric',
  Cost = 'Cost',
  Ignore = 'Ignore',
}

const columnRoleOptions = [
  { label: 'Geo', value: ColumnRole.Geo },
  { label: 'Date', value: ColumnRole.Date },
  { label: 'Metric', value: ColumnRole.Metric },
  { label: 'Cost', value: ColumnRole.Cost },
  { label: 'Ignore', value: ColumnRole.Ignore },
];

const dataPreviewColumns = computed(() => {
  return columns.value.map((col) => ({
    ...col,
    sortable: true,
  }));
});

const uniqueGeoCount = computed(() => {
  if (!currentDataSource.value.columns.geoColumn || rawData.value.length === 0) return 0;
  const uniqueGeos = new Set(
    rawData.value.map((row) => row[currentDataSource.value.columns.geoColumn]),
  );
  return uniqueGeos.size;
});

const uniqueDateCount = computed(() => {
  if (!currentDataSource.value.columns.dateColumn || rawData.value.length === 0) return 0;
  const uniqueDates = new Set(
    rawData.value.map((row) => row[currentDataSource.value.columns.dateColumn]),
  );
  return uniqueDates.size;
});

const dataQualityIssues = computed(() => {
  const issues = [] as string[];

  if (rawData.value.length === 0) return issues;

  // Check for missing geo values
  if (
    currentDataSource.value.columns.geoColumn &&
    rawData.value.some((row) => !row[currentDataSource.value.columns.geoColumn])
  ) {
    issues.push('Some rows have missing values in the geo column');
  }

  // Check for missing date values
  if (
    currentDataSource.value.columns.dateColumn &&
    rawData.value.some((row) => !row[currentDataSource.value.columns.dateColumn])
  ) {
    issues.push('Some rows have missing values in the date column');
  }

  // Check for missing metric values
  if (currentDataSource.value.columns.metricColumns.length > 0) {
    currentDataSource.value.columns.metricColumns.forEach((metric) => {
      if (rawData.value.some((row) => row[metric] === undefined || row[metric] === null)) {
        issues.push(`Column "${metric}" has missing values`);
      }
    });
  }

  return issues;
});

const canContinueFromImport = computed(() => {
  return rawData.value.length > 0;
});

const isColumnConfigValid = computed(() => {
  return (
    !!currentDataSource.value.columns.geoColumn &&
    !!currentDataSource.value.columns.dateColumn &&
    currentDataSource.value.columns.metricColumns.length > 0
  );
});

const isFormValid = computed(() => {
  return !!currentDataSource.value.name && isColumnConfigValid.value;
});

onMounted(async () => {
  // If editing, load existing data source
  const ds = props.dataSource;
  if (ds) {
    // editing an existing DS
    if (!props.dataSource.data) {
      loading.value = true;
      try {
        ds.data = await dataSourcesStore.loadDataSourceData(props.dataSource);
      } catch (error) {
        showError('Failed to load data source details: ' + error);
      } finally {
        loading.value = false;
      }
    }
    // Create a copy to avoid direct mutation (but won't clone 'data')
    if (ds.data) {
      const data = ds.data;
      ds.data = undefined;
      currentDataSource.value = { ...ds };
      currentDataSource.value.data = data;
      ds.data = data;
    } else {
      currentDataSource.value = { ...ds };
    }

    // If there's data loaded, create columns for the editor
    if (currentDataSource.value.data && currentDataSource.value.data.rawRows?.length > 0) {
      // Use the raw rows directly
      rawData.value = currentDataSource.value.data.rawRows.map((row, index) => ({
        ...row,
        __index: index,
      }));

      // Create column definitions from the first row
      if (rawData.value.length > 0) {
        columns.value = Object.keys(rawData.value[0])
          .filter((key) => key !== '__index')
          .map((key) => ({
            name: key,
            label: key,
            field: key,
          }));
      }

      // Initialize column roles
      columnRoles.value = {};
      columns.value.forEach((col) => {
        columnRoles.value[col.name] = ColumnRole.Ignore;
      });

      // Set roles based on current configuration
      if (currentDataSource.value.columns.geoColumn) {
        columnRoles.value[currentDataSource.value.columns.geoColumn] = ColumnRole.Geo;
      }
      if (currentDataSource.value.columns.dateColumn) {
        columnRoles.value[currentDataSource.value.columns.dateColumn] = ColumnRole.Date;
      }
      if (currentDataSource.value.columns.costColumn) {
        columnRoles.value[currentDataSource.value.columns.costColumn] = ColumnRole.Cost;
      }
      currentDataSource.value.columns.metricColumns.forEach((metric) => {
        columnRoles.value[metric] = ColumnRole.Metric;
      });
    }
  }
});

function showError(message: string) {
  errorMessage.value = message;
  errorDialog.value = true;
}

function processData(data: Record<string, unknown>[]) {
  if (!data || data.length === 0) {
    throw new Error('No data found');
  }

  // Add index for row key
  rawData.value = data.map((row, index) => ({
    ...row,
    __index: index,
  }));

  // Create columns from headers
  const firstRow = data[0];
  columns.value = Object.keys(firstRow)
    .filter((key) => key !== '__index') // Filter out the index key
    .map((key) => ({
      name: key,
      label: key,
      field: key,
    }));

  // Initialize column roles
  columnRoles.value = {};
  columns.value.forEach((col) => {
    columnRoles.value[col.name] = ColumnRole.Ignore;
  });

  // Auto-detect columns
  autoDetectColumns();
}

async function handleFileUpload() {
  if (!file.value) return;

  loading.value = true;

  try {
    const text = await file.value.text();

    // Use csv-parse to parse the CSV file
    const parsePromise = new Promise((resolve, reject) => {
      parse(
        text,
        {
          columns: true, // Auto-generate columns from first row
          skip_empty_lines: true, // Skip empty lines
          cast: true, // Attempt to auto-convert values to native types
          trim: true, // Trim whitespace
        },
        (err, records) => {
          if (err) reject(err);
          else resolve(records);
        },
      );
    });

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parsedData = (await parsePromise) as any[];

    if (!parsedData || parsedData.length === 0) {
      throw new Error('The CSV file appears to be empty or has invalid format');
    }

    // Process the data
    processData(parsedData);

    // Set data source name from file if not already set
    if (!currentDataSource.value.name && file.value.name) {
      currentDataSource.value.name = file.value.name.replace(/\.[^/.]+$/, '');
    }
    currentDataSource.value.sourceLink = file.value.name;

    // Show success notification
    $q.notify({
      type: 'positive',
      message: 'File loaded successfully',
      position: 'top',
      timeout: 2000,
    });
  } catch (error) {
    assertIsError(error);
    showError(error.message || 'Failed to parse CSV file');
    clearFile();
  } finally {
    loading.value = false;
  }
}

function clearFile() {
  file.value = null;
  rawData.value = [];
  columns.value = [];
}

function autoDetectColumns() {
  // Reset previous selections
  Object.keys(columnRoles.value).forEach((colName) => {
    columnRoles.value[colName] = ColumnRole.Ignore;
  });

  const colNames = columns.value.map((col) => col.name.toLowerCase());

  // Try to find geo column
  const geoKeywords = ['geo', 'state', 'dma', 'zipcode', 'country', 'region', 'city', 'location'];
  for (const keyword of geoKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Geo;
      break;
    }
  }

  // Try to find date column
  const dateKeywords = ['date', 'time', 'day', 'week', 'month', 'year', 'period'];
  for (const keyword of dateKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Date;
      break;
    }
  }

  // Try to find cost column
  const costKeywords = ['cost', 'spend', 'budget', 'expense', 'investment'];
  for (const keyword of costKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      columnRoles.value[columns.value[matchIndex].name] = ColumnRole.Cost;
      break;
    }
  }

  // Try to identify conversion columns
  const conversionKeywords = [
    'conversions',
    'impressions',
    'visits',
    'signups',
    'purchases',
    'revenue',
    'sale',
    'metric',
  ];
  for (const col of columns.value) {
    const name = col.name.toLowerCase();
    if (columnRoles.value[col.name] === ColumnRole.Ignore) {
      for (const keyword of conversionKeywords) {
        if (name.includes(keyword)) {
          columnRoles.value[col.name] = ColumnRole.Metric;
          break;
        }
      }
    }
  }

  // If no conversion columns were detected but we have numeric columns, use those
  const hasMetrics = Object.values(columnRoles.value).some((role) => role === ColumnRole.Metric);
  if (!hasMetrics) {
    columns.value.forEach((col) => {
      if (
        columnRoles.value[col.name] === ColumnRole.Ignore &&
        rawData.value.slice(0, 5).every((row) => typeof row[col.name] === 'number')
      ) {
        columnRoles.value[col.name] = ColumnRole.Metric;
      }
    });
  }

  // Update the data source columns based on roles
  handleColumnRoleChange('', ColumnRole.Ignore);
}

async function fetchExternalData() {
  if (!externalSourceUrl.value) {
    showError('Please enter a valid Google Sheets URL');
    return;
  }

  loadingExternalData.value = true;
  try {
    const res = await getApiUi(
      'datasources/preview',
      {
        url: externalSourceUrl.value,
      },
      'Fetching data from external source',
    );

    if (!res || !res.data || !Array.isArray(res.data) || res.data.length === 0) {
      return;
    }

    // Process the data using the common function
    processData(res.data as Record<string, unknown>[]);

    // Auto-set data source name from URL if not set
    if (!currentDataSource.value.name) {
      const url = new URL(externalSourceUrl.value);
      currentDataSource.value.name = url.pathname.split('/').pop() || 'External Data Source';
    }

    // Show success notification
    $q.notify({
      type: 'positive',
      message: 'Successfully loaded data from external source',
    });
  } catch (error) {
    assertIsError(error);
    showError('Failed to load external data: ' + error.message);
    // Reset state on error
    rawData.value = [];
    columns.value = [];
  } finally {
    loadingExternalData.value = false;
  }
}

async function saveDataSource() {
  if (!isFormValid.value) {
    showError('Please complete all required fields');
    return;
  }

  saving.value = true;

  try {
    // if (rawData.value.length > 0) {
    //   const processedData = processRawData();
    //   currentDataSource.value.data = processedData;
    // }
    if (!isEditing.value && rawData.value.length > 0) {
      // Set the data
      currentDataSource.value.data = dataSourcesStore.normalizeRawData(
        rawData.value,
        currentDataSource.value.columns,
      );
    }

    // Save to the store
    const res = await dataSourcesStore.saveDataSource(currentDataSource.value);
    if (!res) return;

    // Emit saved event
    emit('saved', res || currentDataSource.value);
  } catch (error) {
    showError('Failed to save data source: ' + error);
  } finally {
    saving.value = false;
  }
}

/*
function processRawData() {
  // This function transforms the raw data into the format expected by the store
  const { geoColumn, dateColumn, metricColumns, costColumn } = currentDataSource.value.columns;

  // Get unique geo units
  const geoUnits = [...new Set(rawData.value.map((row) => row[geoColumn]))];

  // Get unique dates
  const uniqueDates = [...new Set(rawData.value.map((row) => row[dateColumn]))].sort();

  // Get all metric names
  const metricNames = [...metricColumns];

  // Organize data by geo unit
  const byGeo: Record<string, DataRow[]> = {};

  geoUnits.forEach((geo: string) => {
    const geoRows = rawData.value.filter((row) => row[geoColumn] === geo);

    // Convert to DataRow format and sort by date
    byGeo[geo] = geoRows
      .map((row) => {
        // Create a metrics object with all selected metrics
        const metricsObj: Record<string, number> = {};
        for (const metric of metricColumns) {
          if (row[metric] !== undefined && row[metric] !== null) {
            metricsObj[metric] = Number(row[metric]);
          }
        }

        const dataRow: DataRow = {
          geoUnit: row[geoColumn],
          date: row[dateColumn],
          metrics: metricsObj,
        };

        // Add cost if available
        if (costColumn && row[costColumn] !== undefined) {
          dataRow.cost = Number(row[costColumn]);
        }

        return dataRow;
      })
      .sort((a, b) => {
        if (a.date < b.date) return -1;
        if (a.date > b.date) return 1;
        return 0;
      });
  });

  // Calculate statistics for each metric
  const metrics = metricNames.map((name) => {
    const values: number[] = [];

    // Collect all values for this metric
    Object.values(byGeo).forEach((geoRows) => {
      geoRows.forEach((row) => {
        if (row.metrics[name] !== undefined) {
          values.push(row.metrics[name]);
        }
      });
    });

    if (values.length === 0) return { name, min: 0, max: 0 };

    return {
      name,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  });

  return {
    rawRows: rawData.value, // Keep the original raw data
    byGeo,
    geoUnits,
    uniqueDates,
    metricNames,
    metrics,
    numberOfDays: uniqueDates.length,
  };
}
*/

function onCancel() {
  emit('canceled');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleColumnRoleChange(columnName: string, newRole: any) {
  // Extract the actual role value if an object was passed
  const roleValue = typeof newRole === 'object' && newRole !== null ? newRole.value : newRole;

  // Update the role for the column
  if (columnName) {
    columnRoles.value[columnName] = roleValue;
  }

  // If new role is Geo, Date, or Cost, clear that role from all other columns
  if (
    roleValue === ColumnRole.Geo ||
    roleValue === ColumnRole.Date ||
    roleValue === ColumnRole.Cost
  ) {
    Object.keys(columnRoles.value).forEach((col) => {
      if (col !== columnName && columnRoles.value[col] === roleValue) {
        columnRoles.value[col] = ColumnRole.Ignore;
      }
    });
  }

  // Update the data source columns based on roles
  currentDataSource.value.columns.geoColumn = '';
  currentDataSource.value.columns.dateColumn = '';
  currentDataSource.value.columns.metricColumns = [];
  currentDataSource.value.columns.costColumn = undefined;

  Object.entries(columnRoles.value).forEach(([colName, role]) => {
    switch (role) {
      case ColumnRole.Geo:
        currentDataSource.value.columns.geoColumn = colName;
        break;
      case ColumnRole.Date:
        currentDataSource.value.columns.dateColumn = colName;
        break;
      case ColumnRole.Metric:
        currentDataSource.value.columns.metricColumns.push(colName);
        break;
      case ColumnRole.Cost:
        currentDataSource.value.columns.costColumn = colName;
        break;
    }
  });
}
</script>
