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
  <q-card>
    <q-card-section class="row items-center">
      <div class="text-h6">Data Preview: {{ dataSource.name }}</div>
      <q-space />
      <q-btn icon="close" flat round dense v-close-popup />
    </q-card-section>

    <q-separator class="q-my-sm" />

    <q-card-section>
      <div v-if="loading" class="text-center q-py-lg">
        <q-spinner color="primary" size="3em" />
        <div class="q-mt-md">Loading data...</div>
      </div>

      <template v-else-if="initialized">
        <div class="row q-mb-md">
          <div class="col-12 col-md-3">
            <q-card class="bg-primary text-white q-pa-sm">
              <q-card-section>
                <div class="text-h6">Geo Units</div>
                <div class="text-h4">
                  {{ geoUnits.length }}
                </div>
              </q-card-section>
            </q-card>
          </div>
          <div class="col-12 col-md-3">
            <q-card class="bg-secondary text-white q-pa-sm">
              <q-card-section>
                <div class="text-h6">Time Points</div>
                <div class="text-h4">
                  {{ uniqueDates.length }}
                </div>
              </q-card-section>
            </q-card>
          </div>
          <div class="col-12 col-md-3">
            <q-card class="bg-positive text-white q-pa-sm">
              <q-card-section>
                <div class="text-h6">Total Rows</div>
                <div class="text-h4">
                  {{ rawData.length }}
                </div>
              </q-card-section>
            </q-card>
          </div>
          <div class="col-12 col-md-3">
            <q-card class="bg-info text-white q-pa-sm">
              <q-card-section>
                <div class="text-h6">Metrics</div>
                <div class="text-h4">
                  {{ metricColumns.length }}
                </div>
              </q-card-section>
            </q-card>
          </div>
        </div>

        <!-- Geo and Metric selectors -->
        <div class="row q-mb-md q-col-gutter-md">
          <div class="col-12 col-md-6">
            <q-select
              v-model="selectedGeo"
              :options="geoUnits"
              label="Select Geo Unit"
              filled
              emit-value
              @update:model-value="updateDataTable"
              :disable="geoUnits.length === 0"
            />
          </div>

          <div class="col-12 col-md-6">
            <q-select
              v-model="selectedMetrics"
              :options="metricColumns"
              label="Select Metrics to Display"
              filled
              multiple
              use-chips
              emit-value
              @update:model-value="updateDataTable"
              :disable="metricColumns.length === 0"
            />
          </div>
        </div>

        <!-- Metrics Summary Cards -->
        <div class="row q-mb-md q-col-gutter-md" v-if="selectedMetrics.length > 0">
          <div
            v-for="metric in metricStats.filter((m) => selectedMetrics.includes(m.name))"
            :key="metric.name"
            class="col-12 col-sm-6 col-md-4"
          >
            <q-card bordered>
              <q-card-section>
                <div class="text-h6">{{ metric.name }}</div>
                <div class="row q-col-gutter-sm">
                  <div class="col-4">
                    <div class="text-caption">Min</div>
                    <div class="text-subtitle1">{{ formatNumber(metric.min) }}</div>
                  </div>
                  <div class="col-4">
                    <div class="text-caption">Max</div>
                    <div class="text-subtitle1">{{ formatNumber(metric.max) }}</div>
                  </div>
                </div>
              </q-card-section>
            </q-card>
          </div>
        </div>

        <!-- Data table -->
        <q-table
          v-if="selectedGeo && dataTable.length > 0"
          :rows="dataTable"
          :columns="dataColumns"
          row-key="__rowKey"
          :pagination="{ rowsPerPage: 20 }"
          :filter="dataSearch"
        >
          <template v-slot:top>
            <q-input v-model="dataSearch" placeholder="Search" dense filled>
              <template v-slot:append>
                <q-icon name="search" />
              </template>
            </q-input>
          </template>
        </q-table>

        <div v-else-if="geoUnits.length > 0" class="text-center q-pa-md">
          <p>Select a geo unit to view data</p>
        </div>

        <div v-else class="text-center q-pa-md">
          <p>No geo units found in the data</p>
        </div>
      </template>

      <div v-else class="text-center q-pa-md">
        <p>No data available for this data source</p>
      </div>
    </q-card-section>
  </q-card>
</template>

<script setup lang="ts">
import type { QTableColumn } from 'quasar';
import { useQuasar } from 'quasar';
import { formatNumber } from 'src/helpers/utils';
import type { DataSource } from 'stores/datasources';
import { useDataSourcesStore } from 'stores/datasources';
import { ref, computed, onMounted } from 'vue';

const dataSearch = ref('');
const selectedGeo = ref<string | null>(null);
const selectedMetrics = ref<string[]>([]);
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const dataTable = ref<any[]>([]);
const loading = ref(false);
const initialized = ref(false);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const rawData = ref<any[]>([]);
const geoUnits = ref<string[]>([]);
const uniqueDates = ref<string[]>([]);
const metricColumns = ref<string[]>([]);
const metricStats = ref<{ name: string; min: number; max: number }[]>([]);

const $q = useQuasar();

interface Props {
  dataSource: DataSource;
}
const props = defineProps<Props>();

const dataSourcesStore = useDataSourcesStore();

// Data preview columns
const dataColumns = computed(() => {
  const columns: QTableColumn[] = [
    {
      name: 'date',
      label: 'Date',
      field: (row) => row[props.dataSource.columns.dateColumn],
      sortable: true,
      align: 'left',
    },
  ];

  // Add columns for selected metrics
  selectedMetrics.value.forEach((metricName) => {
    columns.push({
      name: metricName,
      label: metricName,
      field: (row) => row[metricName],
      sortable: true,
      align: 'right',
      format: (val) => formatNumber(val),
    });
  });

  // Add cost column if available
  if (props.dataSource.columns.costColumn) {
    columns.push({
      name: 'cost',
      label: props.dataSource.columns.costColumn,
      field: (row) => row[props.dataSource.columns.costColumn || ''],
      sortable: true,
      align: 'right',
      format: (val) => formatNumber(val),
    });
  }

  return columns;
});

onMounted(async () => {
  await loadData();
});

async function loadData() {
  if (!props.dataSource) return;

  loading.value = true;

  try {
    // If data is not available, load it
    if (!props.dataSource.data?.rawRows || !props.dataSource.data.rawRows.length) {
      await dataSourcesStore.loadDataSourceData(props.dataSource);
    }

    // Process data statistics if available
    if (props.dataSource.data) {
      processDataStats();
    }

    initialized.value = true;
  } catch (error) {
    console.error('Error loading data:', error);
    $q.notify({
      type: 'negative',
      message: 'Failed to load data source data',
      position: 'top',
    });
  } finally {
    loading.value = false;
  }
}

function processDataStats() {
  if (!props.dataSource.data || !props.dataSource.data.rawRows) return;

  // Get raw data
  rawData.value = props.dataSource.data.rawRows;

  // Extract columns from schema
  const { geoColumn, dateColumn, metricColumns: metrics } = props.dataSource.columns;

  // Extract unique geo units
  const geoSet = new Set<string>();
  rawData.value.forEach((row) => {
    if (row[geoColumn] !== undefined && row[geoColumn] !== null) {
      geoSet.add(String(row[geoColumn]));
    }
  });
  geoUnits.value = Array.from(geoSet).sort();

  // Extract unique dates
  const dateSet = new Set<string>();
  rawData.value.forEach((row) => {
    if (row[dateColumn] !== undefined && row[dateColumn] !== null) {
      dateSet.add(String(row[dateColumn]));
    }
  });
  uniqueDates.value = Array.from(dateSet).sort();

  // Set metric columns
  metricColumns.value = metrics;

  // Calculate metric statistics
  metricStats.value = metrics.map((metricName) => {
    const values = rawData.value
      .map((row) => row[metricName])
      .filter((val) => typeof val === 'number');

    if (values.length === 0) {
      return { name: metricName, min: 0, max: 0 };
    }

    return {
      name: metricName,
      min: Math.min(...values),
      max: Math.max(...values),
    };
  });

  // Select initial values if needed
  if (geoUnits.value.length > 0 && !selectedGeo.value) {
    selectedGeo.value = geoUnits.value[0];
  }

  if (metrics.length > 0 && selectedMetrics.value.length === 0) {
    selectedMetrics.value = metrics.slice(0, Math.min(3, metrics.length));
  }

  // Update data table
  updateDataTable();
}

function updateDataTable() {
  if (!props.dataSource.data || !selectedGeo.value) {
    dataTable.value = [];
    return;
  }

  const { geoColumn, dateColumn } = props.dataSource.columns;

  // Filter raw rows for selected geo
  const filteredRows = rawData.value.filter((row) => String(row[geoColumn]) === selectedGeo.value);

  // Sort by date
  const sortedRows = [...filteredRows].sort((a, b) => {
    const dateA = String(a[dateColumn]);
    const dateB = String(b[dateColumn]);
    return dateA < dateB ? -1 : dateA > dateB ? 1 : 0;
  });

  // Add a unique key for each row
  dataTable.value = sortedRows.map((row, index) => ({
    ...row,
    __rowKey: `${selectedGeo.value}-${row[dateColumn]}-${index}`,
  }));
}
</script>
