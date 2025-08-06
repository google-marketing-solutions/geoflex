<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="text-h4 q-mb-md">Experiment Analysis</div>

      <q-card class="q-pa-md">
        <div class="row q-col-gutter-md">
          <!-- Design Selection -->
          <div class="col-12">
            <div class="text-subtitle1 q-mb-sm">Experiment Design</div>
            <q-card v-if="selectedDesign" flat bordered class="q-pa-md">
              <div class="text-subtitle2">{{ selectedDesign.design.design_id }}</div>
              <div class="text-caption">
                Created: {{ new Date(selectedDesign.timestamp).toLocaleString() }} (local time)
              </div>
              <div class="text-caption">Data source: {{ selectedDesign.datasource_name }}</div>
              <div class="text-caption">
                Data time frame: {{ formatDate(selectedDesign.start_date) }} -
                {{ formatDate(selectedDesign.end_date) }}
              </div>
              <div class="text-caption">Methodology: {{ selectedDesign.design.methodology }}</div>
              <div class="text-caption">
                Duration: {{ selectedDesign.design.runtime_weeks }} weeks
              </div>
            </q-card>
            <div v-else>
              <!-- TODO: Add selector for designs from GCS or file upload -->
              <q-banner class="bg-grey-3">
                No design selected. Navigate from the Designs page or upload a design to begin.
              </q-banner>
            </div>
          </div>

          <!-- Datasource Selection -->
          <div class="col-12">
            <div class="text-subtitle1 q-mb-sm">Analysis Data Source</div>
            <q-select
              v-model="selectedDataSource"
              :options="dataSourceOptions"
              option-label="name"
              option-value="id"
              label="Choose a data source for analysis"
              outlined
              :loading="dataSourcesStore.loading"
              @popup-show="loadDataSourcesOnOpen"
              @update:model-value="handleDataSourceChange"
            >
              <template v-slot:option="scope">
                <q-item v-bind="scope.itemProps">
                  <q-item-section>
                    <q-item-label>{{ scope.opt.name }}</q-item-label>
                    <q-item-label caption>{{
                      scope.opt.description || 'No description provided'
                    }}</q-item-label>
                  </q-item-section>
                  <q-item-section side>
                    <q-badge :color="scope.opt.data ? 'data' : 'yellow'"> </q-badge>
                  </q-item-section>
                </q-item>
              </template>
              <template v-slot:no-option>
                <q-item>
                  <q-item-section class="text-grey">
                    {{
                      dataSourcesStore.loading ? 'Loading data sources...' : 'No data sources found'
                    }}
                  </q-item-section>
                </q-item>
              </template>
            </q-select>
          </div>
        </div>

        <!-- Experiment Start Date -->
        <div class="row q-col-gutter-md q-mt-md">
          <div class="col-3">
            <div class="text-subtitle1 q-mb-sm">Experiment Start Date</div>
            <q-input
              filled
              v-model="experimentStartDate"
              mask="date"
              :rules="['date']"
              label="Select the start date of the experiment"
            >
              <template v-slot:append>
                <q-icon name="event" class="cursor-pointer">
                  <q-popup-proxy cover transition-show="scale" transition-hide="scale">
                    <q-date v-model="experimentStartDate">
                      <div class="row items-center justify-end">
                        <q-btn v-close-popup label="Close" color="primary" flat />
                      </div>
                    </q-date>
                  </q-popup-proxy>
                </q-icon>
              </template>
            </q-input>
          </div>
        </div>
        <q-separator class="q-my-md" />

        <div class="row justify-end">
          <q-btn
            label="Run Analysis"
            color="primary"
            :disable="
              !selectedDesign ||
              !selectedDataSource ||
              !experimentStartDate ||
              (validationComponent && !validationComponent.validationPassed)
            "
            @click="runAnalysis"
          />
        </div>
      </q-card>

      <!-- Validation Section -->
      <ValidationComponent
        v-if="validationResult"
        :validation="validationResult"
        ref="validationComponent"
        class="q-mt-md"
      />

      <!-- Results Section -->
      <q-card v-if="analysisResults" class="q-mt-md q-pa-md">
        <div class="text-h6 q-mb-md">Analysis Results</div>
        <div v-if="analysisResults.length > 0">
          <div v-for="(result, index) in analysisResults" :key="index" class="q-mb-lg">
            <div class="text-subtitle1 q-mb-sm">
              Result for Metric:
              <span class="text-weight-bold">{{ result.metric }}</span>
              (Cell: {{ result.cell }})
            </div>
            <q-markup-table flat bordered dense>
              <tbody>
                <tr v-for="(value, key) in result" :key="key">
                  <td class="text-weight-medium" style="width: 30%">
                    {{ formatKey(key) }}
                  </td>
                  <td>{{ formatValue(value) }}</td>
                </tr>
              </tbody>
            </q-markup-table>
          </div>
        </div>
        <div v-else>
          <q-banner class="bg-info text-white">
            Analysis completed successfully, but returned no result rows.
          </q-banner>
        </div>
      </q-card>
      <log-viewer v-if="analysisLogs.length" :logs="analysisLogs" class="q-mt-md" />
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useRoute } from 'vue-router';
import { useDataSourcesStore, type DataSource } from 'src/stores/datasources';
import type { LogEntry, SavedDesign, AnyMetric, ValidationResult } from 'src/components/models';
import { postApiUi, getApiUi } from 'boot/axios';
import LogViewer from 'src/components/LogViewer.vue';
import ValidationComponent from 'src/components/ValidationComponent.vue';
import { formatDate } from 'src/helpers/utils';

const formatKey = (key: string | number) => {
  const keyAsString = String(key);
  if (!keyAsString) return '';
  const result = keyAsString.replace(/_/g, ' ');
  return result.charAt(0).toUpperCase() + result.slice(1);
};

const formatValue = (value: string | number | boolean | null) => {
  if (value === null) {
    return 'N/A';
  }
  if (typeof value === 'number') {
    return value.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 4,
    });
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  return value;
};
const route = useRoute();
const dataSourcesStore = useDataSourcesStore();

interface AnalysisResult {
  [key: string]: string | number | boolean | null;
}

const selectedDesign = ref<SavedDesign | null>(null);
const selectedDataSource = ref<DataSource | null>(null);
const experimentStartDate = ref<string | null>(null);
const analysisResults = ref<AnalysisResult[] | null>(null);
const analysisLogs = ref<LogEntry[]>([]);
const dataSourceLoaded = ref(false);
const dataSourceOptions = computed(() => dataSourcesStore.datasources);
const validationResult = ref<ValidationResult | null>(null);
const validationComponent = ref<InstanceType<typeof ValidationComponent> | null>(null);

// Function to load data sources when dropdown is opened
async function loadDataSourcesOnOpen() {
  if (!dataSourcesStore.isLoaded && !dataSourcesStore.loading) {
    await dataSourcesStore.loadDataSources();
  }
}

// Data source handling
const handleDataSourceChange = async (dataSource: DataSource) => {
  if (!dataSource) {
    dataSourceLoaded.value = false;
    return;
  }

  try {
    if (!dataSource.data) {
      await dataSourcesStore.loadDataSourceData(dataSource);
    }
    validateDataSource();
    dataSourceLoaded.value = true;
  } catch (error) {
    console.error('Error loading data source:', error);
  }
};

// Watch for changes in selected design or data source to re-run validation
watch([selectedDesign, selectedDataSource], () => {
  if (selectedDesign.value && selectedDataSource.value) {
    validateDataSource();
  } else {
    validationResult.value = null;
  }
});

function getMetricColumns(metric: AnyMetric): string[] {
  if (typeof metric === 'string') {
    return [metric];
  }
  const columns: string[] = [];
  if (metric.type === 'iroas' && metric.return_column) {
    columns.push(metric.return_column);
  } else if (metric.type === 'cpia' && metric.conversions_column) {
    columns.push(metric.conversions_column);
  } else {
    columns.push(metric.column || metric.name);
  }
  if ((metric.metric_per_cost || metric.cost_per_metric) && metric.cost_column) {
    columns.push(metric.cost_column);
  }
  return columns;
}

function validateDataSource() {
  if (!selectedDesign.value || !selectedDataSource.value || !selectedDataSource.value.data) {
    validationResult.value = null;
    return;
  }

  const design = selectedDesign.value.design;
  const dataSource = selectedDataSource.value;

  // 1. Validate Metrics
  const designMetrics = [
    ...getMetricColumns(design.primary_metric),
    ...design.secondary_metrics.flatMap(getMetricColumns),
  ];
  const dataSourceMetrics = dataSource.data.metricNames || [];
  const missingMetrics = designMetrics.filter((m) => !dataSourceMetrics.includes(m));

  // 2. Validate Geo Units
  const designGeos = new Set([
    ...design.geo_assignment.control,
    ...design.geo_assignment.treatment.flat(),
  ]);
  const dataSourceGeos = new Set(dataSource.data.geoUnits || []);

  const designOnlyGeos = [...designGeos].filter((geo) => !dataSourceGeos.has(geo));
  const dataSourceOnlyGeos = [...dataSourceGeos].filter((geo) => !designGeos.has(geo));

  validationResult.value = {
    metrics: {
      missing: missingMetrics,
    },
    geoUnits: {
      designOnly: designOnlyGeos,
      dataSourceOnly: dataSourceOnlyGeos,
    },
  };
}

// Pre-fill design from route params if available
onMounted(async () => {
  if (route.params.designId) {
    await fetchDesign(route.params.designId as string);
  }
});

async function fetchDesign(designId: string) {
  const response = await getApiUi<SavedDesign>(`designs/${designId}`, {}, 'Loading design');
  if (response && response.data) {
    selectedDesign.value = response.data;
  }
}

async function runAnalysis() {
  if (!selectedDesign.value || !selectedDataSource.value || !experimentStartDate.value) {
    return;
  }

  analysisResults.value = null;

  const payload = {
    datasource_id: selectedDataSource.value.id,
    design_id: selectedDesign.value.design.design_id,
    experiment_start_date: experimentStartDate.value.replace(/\//g, '-'), // API expects YYYY-MM-DD
  };

  const response = await postApiUi<{
    results: AnalysisResult[];
    logs: LogEntry[];
  }>('experiments/analyze', payload, 'Running analysis...');

  if (response) {
    analysisResults.value = response.data.results;
    analysisLogs.value = response.data.logs;
  }
}
</script>
