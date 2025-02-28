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
                  v-model="dataSource.name"
                  label="Data Source Name *"
                  filled
                  :rules="[(val) => !!val || 'Name is required']"
                />
              </div>

              <div class="col-12 col-md-6">
                <q-select
                  v-model="dataSource.sourceType"
                  :options="sourceTypeOptions"
                  label="Data Source Type *"
                  filled
                  emit-value
                  map-options
                />
              </div>

              <div class="col-12">
                <q-input
                  v-model="dataSource.description"
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
          <q-btn color="primary" label="Continue" @click="step = 2" :disable="!dataSource.name" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 2: Data Import -->
      <q-step :name="2" title="Data Import" icon="cloud_upload" :done="step > 2">
        <q-card flat bordered>
          <q-card-section>
            <div v-if="dataSource.sourceType === DataSourceType.Internal">
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
              <p class="text-body1 q-mb-md">Configure connection to an external data source.</p>

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
                label="Test Connection"
                class="q-mt-md"
                @click="testExternalConnection"
                :disable="!externalSourceUrl"
                :loading="testingConnection"
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

      <!-- Step 3: Column Configuration -->
      <q-step :name="3" title="Column Configuration" icon="tune" :done="step > 3">
        <q-card flat bordered>
          <q-card-section>
            <p class="text-body1 q-mb-md">
              Configure which columns represent geo units, date units, and metrics.
            </p>

            <div v-if="dataPreview.length && columns.length">
              <div class="row q-col-gutter-md q-mb-lg">
                <div class="col-12 col-md-6">
                  <q-select
                    v-model="dataSource.geoColumn"
                    :options="columnOptions"
                    label="Geo Unit Column *"
                    filled
                    emit-value
                    map-options
                    :rules="[(val) => !!val || 'Geo column is required']"
                  >
                    <template v-slot:hint>
                      Column containing geographic identifiers (state, DMA, zipcode, etc.)
                    </template>
                  </q-select>
                </div>

                <div class="col-12 col-md-6">
                  <q-select
                    v-model="dataSource.dateColumn"
                    :options="columnOptions"
                    label="Date Unit Column *"
                    filled
                    emit-value
                    map-options
                    :rules="[(val) => !!val || 'Date column is required']"
                  >
                    <template v-slot:hint>
                      Column containing time points (date, week, month, etc.)
                    </template>
                  </q-select>
                </div>
              </div>

              <div class="q-mb-lg">
                <div class="text-subtitle1 q-mb-sm">Metric Columns *</div>
                <p class="text-caption q-mb-md">
                  Select columns that contain metrics you want to analyze
                </p>

                <div class="row q-col-gutter-md">
                  <div v-for="col in columns" :key="col.name" class="col-12 col-md-4 col-lg-3">
                    <q-checkbox v-model="metricColumns" :val="col.name" :label="col.name" />
                  </div>
                </div>
                <div v-if="metricColumns.length === 0" class="text-negative text-caption q-mt-sm">
                  At least one metric column is required
                </div>
              </div>

              <div>
                <div class="text-subtitle1 q-mb-sm">Cost Column (optional)</div>
                <p class="text-caption q-mb-md">
                  Select a column that contains cost data, if available
                </p>

                <q-select
                  v-model="dataSource.costColumn"
                  :options="columnOptions"
                  label="Cost Column"
                  filled
                  emit-value
                  map-options
                  clearable
                >
                  <template v-slot:hint> Column containing cost or budget information </template>
                </q-select>
              </div>
            </div>

            <div v-else class="text-center">
              <p>No data preview available. Please upload data in the previous step.</p>
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
          <q-btn flat color="primary" label="Back" class="q-mr-sm" @click="step = 2" />
          <q-btn flat color="primary" label="Cancel" class="q-ml-sm" @click="onCancel" />
        </q-stepper-navigation>
      </q-step>

      <!-- Step 4: Data Preview -->
      <q-step :name="4" title="Data Preview" icon="preview">
        <q-card flat bordered>
          <q-card-section>
            <div v-if="dataPreview.length && columns.length">
              <div class="text-h6 q-mb-sm">Data Preview</div>
              <p class="text-caption q-mb-md">
                Preview of the first {{ dataPreview.length }} rows from your data source.
              </p>

              <q-table
                :rows="dataPreview"
                :columns="tableColumns"
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

            <div v-else class="text-center">
              <p>No data preview available. Please upload data in the previous steps.</p>
            </div>
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
import { ref, computed, watch, onMounted } from 'vue';
import { useQuasar } from 'quasar';
import { parse } from 'csv-parse/browser/esm';
import { useDataSourcesStore, DataSourceType, DataSource, DataRow } from 'stores/datasources';
import { v4 as uuidv4 } from 'uuid';

// Props and emits
interface Props {
  editingId?: string | null;
}

const props = withDefaults(defineProps<Props>(), {
  editingId: null,
});

const emit = defineEmits<{
  (e: 'saved'): void;
  (e: 'canceled'): void;
}>();

// Quasar utilities
const $q = useQuasar();

// Store
const dataSourcesStore = useDataSourcesStore();

// Step management
const step = ref(1);

// Form data
const dataSource = ref<DataSource>({
  id: '',
  name: '',
  description: '',
  sourceType: DataSourceType.Internal,
  geoColumn: '',
  dateColumn: '',
  metricColumns: [],
  costColumn: '',
});

// UI state
const file = ref<File | null>(null);
const loading = ref(false);
const saving = ref(false);
const errorDialog = ref(false);
const errorMessage = ref('');
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const dataPreview = ref<any[]>([]);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const rawData = ref<any[]>([]);
const columns = ref<{ name: string; label: string; field: string }[]>([]);
const metricColumns = ref<string[]>([]);
const search = ref('');
const testingConnection = ref(false);
const externalSourceUrl = ref('');

// Select options
const sourceTypeOptions = [
  { label: 'Internal (Upload CSV)', value: DataSourceType.Internal },
  { label: 'External (Google Sheets, etc.)', value: DataSourceType.External },
];

// Computed properties
const columnOptions = computed(() => {
  return columns.value.map((col) => ({
    label: col.label,
    value: col.name,
  }));
});

const tableColumns = computed(() => {
  return columns.value.map((col) => ({
    ...col,
    sortable: true,
  }));
});

const uniqueGeoCount = computed(() => {
  if (!dataSource.value.geoColumn || rawData.value.length === 0) return 0;
  const uniqueGeos = new Set(rawData.value.map((row) => row[dataSource.value.geoColumn]));
  return uniqueGeos.size;
});

const uniqueDateCount = computed(() => {
  if (!dataSource.value.dateColumn || rawData.value.length === 0) return 0;
  const uniqueDates = new Set(rawData.value.map((row) => row[dataSource.value.dateColumn]));
  return uniqueDates.size;
});

const dataQualityIssues = computed(() => {
  const issues = [] as string[];

  if (rawData.value.length === 0) return issues;

  // Check for missing geo values
  if (dataSource.value.geoColumn && rawData.value.some((row) => !row[dataSource.value.geoColumn])) {
    issues.push('Some rows have missing values in the geo column');
  }

  // Check for missing date values
  if (
    dataSource.value.dateColumn &&
    rawData.value.some((row) => !row[dataSource.value.dateColumn])
  ) {
    issues.push('Some rows have missing values in the date column');
  }

  // Check for missing metric values
  if (metricColumns.value.length > 0) {
    metricColumns.value.forEach((metric) => {
      if (rawData.value.some((row) => row[metric] === undefined || row[metric] === null)) {
        issues.push(`Column "${metric}" has missing values`);
      }
    });
  }

  return issues;
});

const canContinueFromImport = computed(() => {
  if (dataSource.value.sourceType === DataSourceType.Internal) {
    return rawData.value.length > 0;
  } else {
    return !!externalSourceUrl.value;
  }
});

const isColumnConfigValid = computed(() => {
  return (
    !!dataSource.value.geoColumn && !!dataSource.value.dateColumn && metricColumns.value.length > 0
  );
});

const isFormValid = computed(() => {
  return !!dataSource.value.name && isColumnConfigValid.value;
});

// Watchers
watch(metricColumns, (newValue) => {
  dataSource.value.metricColumns = [...newValue];
});

// Methods
onMounted(async () => {
  // If editing, load existing data source
  if (props.editingId) {
    loading.value = true;
    try {
      // Get the data source from the store
      const existingSource = dataSourcesStore.getDataSourceById(props.editingId);

      if (existingSource) {
        // Create a copy to avoid direct mutation
        dataSource.value = { ...existingSource };

        // Set metricColumns ref to match the data source
        metricColumns.value = Array.isArray(dataSource.value.metricColumns)
          ? [...dataSource.value.metricColumns]
          : [];

        // If external source, set the external URL
        if (dataSource.value.sourceType === DataSourceType.External) {
          externalSourceUrl.value = existingSource.externalSourceUrl || '';
        }

        // If there's data loaded, create columns for the editor
        if (existingSource.data?.rows) {
          // Try to extract sample data
          const sampleGeo = existingSource.data.geo[0];
          if (sampleGeo && existingSource.data.rows[sampleGeo]?.length > 0) {
            // Use the first row as a sample
            //const sampleRow = existingSource.data.rows[sampleGeo][0];

            // Create column definitions
            columns.value = [
              {
                name: dataSource.value.geoColumn,
                label: dataSource.value.geoColumn,
                field: dataSource.value.geoColumn,
              },
              {
                name: dataSource.value.dateColumn,
                label: dataSource.value.dateColumn,
                field: dataSource.value.dateColumn,
              },
              // Add metric columns
              ...dataSource.value.metricColumns.map((col) => ({
                name: col,
                label: col,
                field: col,
              })),
              // Add cost column if present
              ...(dataSource.value.costColumn
                ? [
                    {
                      name: dataSource.value.costColumn,
                      label: dataSource.value.costColumn,
                      field: dataSource.value.costColumn,
                    },
                  ]
                : []),
            ];

            // Create sample data preview
            const previewData = existingSource.data.rows[sampleGeo]
              .slice(0, 10)
              .map((row, index) => ({
                ...row,
                __index: index,
                [dataSource.value.geoColumn]: row.geoUnit,
                [dataSource.value.dateColumn]: row.date,
                [dataSource.value.metricColumns[0]]: row.metric,
                ...(dataSource.value.costColumn ? { [dataSource.value.costColumn]: row.cost } : {}),
              }));

            dataPreview.value = previewData;
            rawData.value = [...previewData];
          }
        }
      }
    } catch (error) {
      showError('Failed to load data source details');
    } finally {
      loading.value = false;
    }
  } else {
    // For new data source, set default ID
    dataSource.value.id = uuidv4();
  }
});

function showError(message: string) {
  errorMessage.value = message;
  errorDialog.value = true;
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

    // Add index for row key
    rawData.value = parsedData.map((row, index) => ({
      ...row,
      __index: index,
    }));

    // Show preview (first N rows)
    dataPreview.value = rawData.value.slice(0, 10);

    // Create columns from the CSV headers
    const firstRow = parsedData[0];
    columns.value = Object.keys(firstRow).map((key) => ({
      name: key,
      label: key,
      field: key,
    }));

    // Auto-detect columns
    autoDetectColumns();

    // Set data source name from file if not already set
    if (!dataSource.value.name && file.value.name) {
      dataSource.value.name = file.value.name.replace(/\.[^/.]+$/, '');
    }

    // Show success notification
    $q.notify({
      type: 'positive',
      message: 'File loaded successfully',
      position: 'top',
      timeout: 2000,
    });
  } catch (error) {
    showError(error instanceof Error ? error.message : 'Failed to parse CSV file');
    clearFile();
  } finally {
    loading.value = false;
  }
}

function clearFile() {
  file.value = null;
  dataPreview.value = [];
  rawData.value = [];
  columns.value = [];
  metricColumns.value = [];
}

function autoDetectColumns() {
  // Reset previous selections
  dataSource.value.geoColumn = '';
  dataSource.value.dateColumn = '';
  metricColumns.value = [];
  dataSource.value.costColumn = '';

  const colNames = columns.value.map((col) => col.name.toLowerCase());

  // Try to find geo column
  const geoKeywords = ['geo', 'state', 'dma', 'zipcode', 'country', 'region', 'city', 'location'];
  for (const keyword of geoKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      dataSource.value.geoColumn = columns.value[matchIndex].name;
      break;
    }
  }

  // Try to find date column
  const dateKeywords = ['date', 'time', 'day', 'week', 'month', 'year', 'period'];
  for (const keyword of dateKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      dataSource.value.dateColumn = columns.value[matchIndex].name;
      break;
    }
  }

  // Try to find cost column
  const costKeywords = ['cost', 'spend', 'budget', 'expense', 'investment'];
  for (const keyword of costKeywords) {
    const matchIndex = colNames.findIndex((name) => name.includes(keyword));
    if (matchIndex !== -1) {
      dataSource.value.costColumn = columns.value[matchIndex].name;
      break;
    }
  }

  // Try to identify conversion columns
  const conversionKeywords = [
    'conversion',
    'visit',
    'signup',
    'purchase',
    'revenue',
    'sale',
    'metric',
  ];
  for (const col of columns.value) {
    const name = col.name.toLowerCase();
    if (
      name !== dataSource.value.geoColumn.toLowerCase() &&
      name !== dataSource.value.dateColumn.toLowerCase() &&
      name !== dataSource.value.costColumn.toLowerCase()
    ) {
      for (const keyword of conversionKeywords) {
        if (name.includes(keyword)) {
          metricColumns.value.push(col.name);
          break;
        }
      }
    }
  }

  // If no conversion columns were detected but we have numeric columns, use those
  if (metricColumns.value.length === 0) {
    const numericColumns = columns.value.filter((col) => {
      // Skip geo, date, and cost columns
      if (
        col.name === dataSource.value.geoColumn ||
        col.name === dataSource.value.dateColumn ||
        col.name === dataSource.value.costColumn
      ) {
        return false;
      }

      // Check first few rows to see if column contains numeric values
      return dataPreview.value.slice(0, 5).every((row) => typeof row[col.name] === 'number');
    });

    metricColumns.value = numericColumns.map((col) => col.name);
  }
}

async function testExternalConnection() {
  if (!externalSourceUrl.value) return;

  testingConnection.value = true;

  try {
    // In a real app, you would test the connection to the external source
    // For now, we'll just simulate a delay
    await new Promise((resolve) => setTimeout(resolve, 1500));

    // Store external source URL in the data source
    dataSource.value.externalSourceUrl = externalSourceUrl.value;

    // Simulate success
    $q.notify({
      type: 'positive',
      message: 'Connection successful',
      position: 'top',
      timeout: 2000,
    });

    // Simulate some sample data
    rawData.value = Array.from({ length: 50 }, (_, i) => ({
      __index: i,
      State: ['NY', 'CA', 'TX', 'FL', 'IL'][Math.floor(Math.random() * 5)],
      Date: `2023-${String(Math.floor(Math.random() * 12) + 1).padStart(2, '0')}-${String(Math.floor(Math.random() * 28) + 1).padStart(2, '0')}`,
      Visits: Math.floor(Math.random() * 1000) + 100,
      Conversions: Math.floor(Math.random() * 100) + 10,
      Cost: Math.floor(Math.random() * 500) + 50,
    }));

    dataPreview.value = rawData.value.slice(0, 10);

    columns.value = [
      { name: 'State', label: 'State', field: 'State' },
      { name: 'Date', label: 'Date', field: 'Date' },
      { name: 'Visits', label: 'Visits', field: 'Visits' },
      { name: 'Conversions', label: 'Conversions', field: 'Conversions' },
      { name: 'Cost', label: 'Cost', field: 'Cost' },
    ];

    // Auto-detect columns
    autoDetectColumns();
  } catch (error) {
    showError('Failed to connect to external source');
  } finally {
    testingConnection.value = false;
  }
}

async function saveDataSource() {
  if (!isFormValid.value) {
    showError('Please complete all required fields');
    return;
  }

  saving.value = true;

  try {
    // Update data source with current metric columns
    dataSource.value.metricColumns = [...metricColumns.value];

    // Add external source URL if applicable
    if (dataSource.value.sourceType === DataSourceType.External) {
      dataSource.value.externalSourceUrl = externalSourceUrl.value;
    }

    // For internal sources, process and include the data
    if (dataSource.value.sourceType === DataSourceType.Internal && rawData.value.length > 0) {
      const processedData = processRawData();
      dataSource.value.data = processedData;
    }

    // Save to the store
    await dataSourcesStore.saveDataSource(dataSource.value);

    // Emit saved event
    emit('saved');
  } catch (error) {
    showError('Failed to save data source');
  } finally {
    saving.value = false;
  }
}

function processRawData() {
  // This function transforms the raw data into the format expected by the store
  const { geoColumn, dateColumn, metricColumns, costColumn } = dataSource.value;

  // Get unique geo units
  const geoUnits = [...new Set(rawData.value.map((row) => row[geoColumn]))];

  // Organize data by geo unit
  const organizedData: Record<string, DataRow[]> = {};

  geoUnits.forEach((geo: string) => {
    const geoRows = rawData.value.filter((row) => row[geoColumn] === geo);

    // Convert to DataRow format and sort by date
    organizedData[geo] = geoRows
      .map((row) => ({
        geoUnit: row[geoColumn],
        date: row[dateColumn],
        // For simplicity, using the first metric column
        metric: row[metricColumns[0]],
        cost: costColumn ? row[costColumn] : undefined,
      }))
      .sort((a, b) => {
        if (a.date < b.date) return -1;
        if (a.date > b.date) return 1;
        return 0;
      });
  });

  // Count unique dates
  const allDates = rawData.value.map((row) => row[dateColumn]);
  const uniqueDates = [...new Set(allDates)];

  return {
    rows: organizedData,
    geo: geoUnits,
    numberOfDays: uniqueDates.length,
  };
}

function onCancel() {
  emit('canceled');
}
</script>
