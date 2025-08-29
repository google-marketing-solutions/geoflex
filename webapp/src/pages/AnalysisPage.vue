<!--
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="row justify-between items-center q-mb-md">
        <div>
          <div class="text-h4">GeoFlex Experiment Report</div>
          <div v-if="selectedDesign" class="text-subtitle1 text-grey-7">
            {{ selectedDesign.design.design_id }}
          </div>
        </div>
        <div>
          <q-btn
            v-if="analysisResults"
            icon="download"
            label="Download Report"
            color="primary"
            unelevated
            @click="downloadReport"
          />
        </div>
      </div>

      <q-card class="q-pa-md">
        <div class="row q-col-gutter-md" v-if="!analysisResults">
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

        <!-- Experiment Start/End Dates -->
        <div class="row q-col-gutter-md q-mt-md" v-if="!analysisResults">
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
                  <q-popup-proxy
                    ref="startDateProxy"
                    cover
                    transition-show="scale"
                    transition-hide="scale"
                  >
                    <q-date
                      v-model="experimentStartDate"
                      @update:model-value="() => startDateProxy.hide()"
                    />
                  </q-popup-proxy>
                </q-icon>
              </template>
            </q-input>
          </div>
          <div class="col-3">
            <div class="text-subtitle1 q-mb-sm">Experiment End Date</div>
            <q-input
              filled
              clearable
              v-model="experimentEndDate"
              mask="date"
              label="Select the end date of the experiment"
              hint="By default it is determined by adding runtime weeks from the design to start date"
            >
              <template v-slot:append>
                <q-icon name="event" class="cursor-pointer">
                  <q-popup-proxy
                    ref="endDateProxy"
                    cover
                    transition-show="scale"
                    transition-hide="scale"
                  >
                    <q-date
                      v-model="experimentEndDate"
                      @update:model-value="() => endDateProxy.hide()"
                    />
                  </q-popup-proxy>
                </q-icon>
              </template>
            </q-input>
          </div>
        </div>

        <!-- Metric Results -->
        <div class="row q-mt-md" v-if="analysisResults">
          <div class="col-12 col-md-8 col-lg-9">
            <div v-if="analysisResults.length > 0">
              <div v-for="(result, index) in analysisResults" :key="index" class="q-mb-xl">
                <div class="text-h6 q-mb-md">
                  Metric:
                  <span class="text-weight-bold"
                    >{{ result.metric }} ({{
                      result.is_primary_metric ? 'primary' : 'secondary'
                    }})</span
                  >
                </div>
                <div class="row q-col-gutter-md">
                  <!-- Significance Card -->
                  <div class="col-12 col-sm-6 col-md-3">
                    <q-card class="q-pa-md full-height">
                      <div class="row items-center no-wrap">
                        <q-icon
                          :name="result.is_significant ? 'check_circle' : 'cancel'"
                          :color="result.is_significant ? 'positive' : 'negative'"
                          size="2em"
                          class="q-mr-sm"
                        />
                        <div
                          class="text-h6"
                          :class="result.is_significant ? 'text-positive' : 'text-negative'"
                        >
                          {{
                            result.is_significant ? 'Statistically Significant' : 'Not Significant'
                          }}
                        </div>
                      </div>
                      <div class="q-mt-sm text-grey-7">
                        <div>
                          p-value:
                          {{ formatValue(result.p_value, 'number', 4) }}
                        </div>
                        <!-- <div>95% Confidence Interval</div> -->
                      </div>
                    </q-card>
                  </div>

                  <!-- Incremental Lift Card -->
                  <div class="col-12 col-sm-6 col-md-3">
                    <q-card class="q-pa-md full-height">
                      <div class="text-subtitle1 text-grey-7">Incremental Lift</div>
                      <div class="text-h4 q-mt-xs">
                        {{ formatValue(result.point_estimate_relative, 'percent') }}
                      </div>
                      <!-- <div class="text-grey-7 q-mt-sm">
                        Increase in {{ result.metric }} in test geos.
                      </div> -->
                    </q-card>
                  </div>

                  <!-- Total Incremental Conversions Card -->
                  <div class="col-12 col-sm-6 col-md-3">
                    <q-card class="q-pa-md full-height">
                      <div class="text-subtitle1 text-grey-7">
                        Total Incremental {{ result.metric }}
                      </div>
                      <div class="text-h4 q-mt-xs">
                        {{ formatValue(result.point_estimate, 'number', 2) }}
                      </div>
                      <!-- <div class="text-grey-7 q-mt-sm">
                        Additional {{ result.metric }} generated.
                      </div> -->
                    </q-card>
                  </div>

                  <!-- Confidence Interval Card -->
                  <div class="col-12 col-sm-6 col-md-3">
                    <q-card class="q-pa-md full-height">
                      <div class="text-subtitle1 text-grey-7">Absolute Effect</div>
                      <div class="text-h6 q-mt-xs">
                        {{ formatValue(result.lower_bound, 'number', 2) }}
                        &mdash;
                        {{ formatValue(result.upper_bound, 'number', 2) }}
                      </div>
                      <div class="text-grey-7 q-mt-sm">
                        Confidence Interval:<br />
                        [{{ formatValue(result.lower_bound_relative, 'number', 2) }};
                        {{ formatValue(result.upper_bound_relative, 'number', 2) }}]
                      </div>
                    </q-card>
                  </div>
                </div>
              </div>
            </div>
            <div v-else>
              <q-banner class="bg-info text-white">
                Analysis completed successfully, but returned no result rows.
              </q-banner>
            </div>
          </div>
        </div>

        <!-- Results: Experiment Details -->
        <data class="row q-mt-md" v-if="analysisResults">
          <div class="col-4 col-md-4 col-lg-3">
            <q-card class="q-pa-md">
              <div class="text-h6 q-mb-md">Experiment Details</div>
              <div
                v-for="(item, index) in experimentDetails"
                :key="index"
                class="row q-mb-sm items-center"
              >
                <div class="col-6 text-grey-7">{{ item.label }}:</div>
                <div class="col-6 text-weight-medium text-right">
                  <span>{{ item.value }}</span>
                </div>
              </div>
            </q-card>
          </div>
        </data>

        <div class="row q-col-gutter-md q-mt-md">
          <div class="col-12">
            <ExperimentTimeline
              v-if="
                selectedDesign &&
                selectedDataSource &&
                selectedDataSource.data &&
                experimentStartDate
              "
              :design="selectedDesign"
              :data-source="selectedDataSource"
              :experiment-start-date="experimentStartDate"
              :experiment-end-date="experimentEndDate"
            />
          </div>
        </div>

        <!-- Validation Section -->
        <ValidationComponent
          v-if="validationResult && !analysisResults"
          :validation="validationResult"
          ref="validationComponent"
          class="q-mt-md"
        />

        <q-separator class="q-my-md" />

        <div class="row justify-end">
          <q-btn
            v-if="!analysisResults"
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
          <q-btn
            v-if="analysisResults"
            icon="edit"
            label="Modify Parameters"
            color="secondary"
            outline
            class="q-mr-sm"
            @click="analysisResults = null"
          />
        </div>
      </q-card>

      <!-- Results Section -->
      <div class="q-mt-md"></div>
      <log-viewer v-if="analysisLogs.length" :logs="analysisLogs" class="q-mt-md" />
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import type { QPopupProxy } from 'quasar';
import { useRoute } from 'vue-router';
import { useDataSourcesStore, type DataSource } from 'src/stores/datasources';
import type { LogEntry, SavedDesign, AnyMetric, ValidationResult } from 'src/components/models';
import { postApiUi, getApiUi } from 'boot/axios';
import LogViewer from 'src/components/LogViewer.vue';
import ValidationComponent from 'src/components/ValidationComponent.vue';
import ExperimentTimeline from 'src/components/ExperimentTimeline.vue';
import { formatDate } from 'src/helpers/utils';

const formatValue = (
  value: string | number | boolean | null,
  type: 'string' | 'number' | 'percent' | 'boolean' = 'string',
  maxFractionDigits = 4,
) => {
  if (value === null || value === undefined) {
    return 'N/A';
  }

  if (type === 'percent') {
    const numValue = typeof value === 'string' ? parseFloat(value) : (value as number);
    if (isNaN(numValue)) return 'N/A';
    return `${(numValue * 100).toFixed(1)}%`;
  }

  if (type === 'number' || typeof value === 'number') {
    const numValue = typeof value === 'string' ? parseFloat(value) : (value as number);
    if (isNaN(numValue)) return 'N/A';
    return numValue.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: maxFractionDigits,
    });
  }

  if (type === 'boolean' || typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }

  return String(value);
};
const route = useRoute();
const dataSourcesStore = useDataSourcesStore();

interface AnalysisResult {
  [key: string]: string | number | boolean | null;
}

const selectedDesign = ref<SavedDesign | null>(null);
const selectedDataSource = ref<DataSource | null>(null);
const experimentStartDate = ref<string | null>(null);
const experimentEndDate = ref<string | null>(null);
const analysisResults = ref<AnalysisResult[] | null>(null);
const analysisLogs = ref<LogEntry[]>([]);
const dataSourceLoaded = ref(false);
const dataSourceOptions = computed(() => dataSourcesStore.datasources);
const validationResult = ref<ValidationResult | null>(null);
const validationComponent = ref<InstanceType<typeof ValidationComponent> | null>(null);
const startDateProxy = ref<QPopupProxy | null>(null);
const endDateProxy = ref<QPopupProxy | null>(null);

const experimentDetails = computed(() => {
  if (!selectedDesign.value) return [];
  const design = selectedDesign.value.design;

  return [
    {
      label: 'Test Duration',
      value: `${design.runtime_weeks} Weeks`,
    },
    {
      label: 'Test Type',
      value: design.effect_scope,
    },
    {
      label: 'Methodology',
      value: design.methodology,
    },
    {
      label: 'Test Geos',
      value: design.geo_assignment.treatment.flat().length,
    },
    {
      label: 'Control Geos',
      value: design.geo_assignment.control.length,
    },
  ];
});

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
watch([selectedDesign, selectedDataSource, experimentStartDate, experimentEndDate], () => {
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

  // 3. Validate Dates
  let datesValidation = null;
  if (experimentStartDate.value && dataSource.data?.uniqueDates) {
    const expStartDate = new Date(experimentStartDate.value);
    const expEndDate = experimentEndDate.value
      ? new Date(experimentEndDate.value)
      : new Date(expStartDate.getTime() + design.runtime_weeks * 7 * 24 * 60 * 60 * 1000);

    const dsStartDate = new Date(dataSource.data.uniqueDates[0]);
    const dsEndDate = new Date(dataSource.data.uniqueDates[dataSource.data.uniqueDates.length - 1]);

    const valid = expStartDate >= dsStartDate && expEndDate <= dsEndDate;

    datesValidation = {
      valid,
      required: {
        start: expStartDate.toISOString().split('T')[0],
        end: expEndDate.toISOString().split('T')[0],
      },
      actual: {
        start: dsStartDate.toISOString().split('T')[0],
        end: dsEndDate.toISOString().split('T')[0],
      },
    };
  }

  validationResult.value = {
    metrics: {
      missing: missingMetrics,
    },
    geoUnits: {
      designOnly: designOnlyGeos,
      dataSourceOnly: dataSourceOnlyGeos,
    },
    dates: datesValidation,
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

const downloadReport = () => {
  // TODO: Implement report download functionality
  alert('Report download functionality is not yet implemented.');
};

async function runAnalysis() {
  if (!selectedDesign.value || !selectedDataSource.value || !experimentStartDate.value) {
    return;
  }

  analysisResults.value = null;

  const payload = {
    datasource_id: selectedDataSource.value.id,
    design_id: selectedDesign.value.design.design_id,
    experiment_start_date: experimentStartDate.value.replace(/\//g, '-'), // API expects YYYY-MM-DD
    experiment_end_date: experimentEndDate.value?.replace(/\//g, '-'),
  };

  const response = await postApiUi<{
    results: AnalysisResult[];
    logs: LogEntry[];
  }>('experiments/analyze', payload, 'Running analysis...');

  if (response) {
    analysisResults.value = response.data.results;
    analysisLogs.value = response.data.logs;

    /**
     "metric": "revenue",
     "cell": 1,
     "point_estimate": 13.928580899098364,
     "lower_bound": -109.35050997899843,
     "upper_bound": 137.20767177719517,
     "p_value": 0.818422545387028,
     "is_significant": false,
     "is_primary_metric": true

     "point_estimate_relative": 0.009646595480609296,
     "lower_bound_relative": -0.1327475286297486,
     "upper_bound_relative": 0.17294733243686844,

"metric", # The name of the metric
"cell",   # The number of the treatment cell
"point_estimate",  # The absolute effect size
"lower_bound",     # Lower bound on the absolute effect size
"upper_bound",     # Upper bound on the absolute effect size
"p_value",         # P-value for the test
"is_significant"   # Is the test stat-sig?

Only available for non cost based metrics:
"point_estimate_relative",  # Relative lift (0.2 = +20%)
"lower_bound_relative",
"upper_bound_relative",
     */
  }
}
</script>
