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

              <div class="q-my-md">
                <q-btn
                  label="Validate Data"
                  color="secondary"
                  @click="triggerValidation"
                  :loading="validateDataLoading"
                  icon="rule"
                />
              </div>

              <div v-if="dataQualityIssues.length > 0" class="q-mt-md">
                <q-banner class="bg-warning text-white">
                  <template v-slot:avatar>
                    <q-icon name="warning" />
                  </template>
                  <div class="text-subtitle1 q-mb-sm">Data Quality Issues</div>
                  <ul class="q-mt-none q-mb-none">
                    <li v-for="(issue, index) in dataQualityIssues" :key="index">
                      {{ issue.message }}
                    </li>
                  </ul>
                  <div class="q-mt-sm">
                    <q-btn
                      label="Attempt to Fix Issues"
                      color="primary"
                      @click="openFixDialog"
                      :loading="isFixingIssues"
                      icon="auto_fix_high"
                    />
                  </div>
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

    <!-- Fix Selection Dialog -->
    <q-dialog v-model="showFixSelectionDialog" persistent>
      <q-card style="min-width: 350px">
        <q-card-section>
          <div class="text-h6">Select Fixes to Apply</div>
        </q-card-section>

        <q-card-section class="q-pt-none">
          <q-option-group
            v-model="fixSelectionArray"
            :options="fixSelectionOptions"
            type="checkbox"
            dense
          />
        </q-card-section>

        <q-card-actions align="right">
          <q-btn flat label="Cancel" color="primary" v-close-popup />
          <q-btn
            label="Apply Selected Fixes"
            color="primary"
            @click="applySelectedFixes"
            :loading="isFixingIssues"
          />
        </q-card-actions>
      </q-card>
    </q-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useQuasar } from 'quasar';

enum DataQualityIssueType {
  MissingGeo = 'MissingGeo',
  MissingDate = 'MissingDate',
  MissingMetric = 'MissingMetric',
  InvalidDateFormat = 'InvalidDateFormat',
  NonNumericMetric = 'NonNumericMetric',
  DateGap = 'DateGap',
  DuplicateGeoDate = 'DuplicateGeoDate',
}

interface DataQualityIssue {
  type: DataQualityIssueType;
  message: string;
  isUnfixableFormatError?: boolean;
}
import { parse } from 'csv-parse/browser/esm';
import type { DataSource } from 'stores/datasources'; // Removed DataRow, DataSourceData
import { useDataSourcesStore, DataSourceType } from 'stores/datasources';
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
const isFixingIssues = ref(false);
const validateDataLoading = ref(false);
const showFixSelectionDialog = ref(false);

interface FixSelection {
  removeMissingCritical: boolean;
  fillMissingMetrics: boolean;
  fillDateGaps: boolean;
  deduplicateRows: boolean;
}
const fixSelection = ref<FixSelection>({
  removeMissingCritical: true,
  fillMissingMetrics: true,
  fillDateGaps: true,
  deduplicateRows: true,
});

// For q-option-group which expects an array for multiple checkboxes
const fixSelectionArray = computed({
  get: () => {
    const arr = [];
    if (fixSelection.value.removeMissingCritical) arr.push('removeMissingCritical');
    if (fixSelection.value.fillMissingMetrics) arr.push('fillMissingMetrics');
    if (fixSelection.value.fillDateGaps) arr.push('fillDateGaps');
    if (fixSelection.value.deduplicateRows) arr.push('deduplicateRows'); // Added deduplicateRows
    return arr;
  },
  set: (val: string[]) => {
    fixSelection.value.removeMissingCritical = val.includes('removeMissingCritical');
    fixSelection.value.fillMissingMetrics = val.includes('fillMissingMetrics');
    fixSelection.value.fillDateGaps = val.includes('fillDateGaps');
    fixSelection.value.deduplicateRows = val.includes('deduplicateRows'); // Added deduplicateRows
  },
});

const availableFixTypes = computed(() => {
  const types = {
    removeMissingCritical: false,
    fillMissingMetrics: false,
    fillDateGaps: false,
    deduplicateRows: false,
  };

  const currentIssues: DataQualityIssue[] = dataQualityIssues.value;

  if (
    currentIssues.some(
      (issue) =>
        issue.type === DataQualityIssueType.MissingGeo ||
        issue.type === DataQualityIssueType.MissingDate,
    )
  ) {
    types.removeMissingCritical = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.MissingMetric)) {
    types.fillMissingMetrics = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.DateGap)) {
    types.fillDateGaps = true;
  }

  if (currentIssues.some((issue) => issue.type === DataQualityIssueType.DuplicateGeoDate)) {
    types.deduplicateRows = true;
  }

  if (currentIssues.some((issue) => issue.isUnfixableFormatError === true)) {
    types.removeMissingCritical = false;
    types.fillMissingMetrics = false;
    types.fillDateGaps = false;
    types.deduplicateRows = false;
  }

  return types;
});

const fixSelectionOptions = computed(() => [
  {
    label: 'Remove rows with missing Geo/Date',
    value: 'removeMissingCritical',
    disable: !availableFixTypes.value.removeMissingCritical,
  },
  {
    label: 'Fill missing metric values with 0',
    value: 'fillMissingMetrics',
    disable: !availableFixTypes.value.fillMissingMetrics,
  },
  {
    label: 'Fill date gaps with 0-value entries',
    value: 'fillDateGaps',
    disable: !availableFixTypes.value.fillDateGaps,
  },
  {
    label: 'Remove duplicate Geo/Date rows',
    value: 'deduplicateRows',
    disable: !availableFixTypes.value.deduplicateRows,
  },
]);

// Select options
const sourceTypeOptions = [
  { label: 'Internal (Upload CSV)', value: DataSourceType.Internal },
  { label: 'External (Google Sheets, etc.)', value: DataSourceType.External },
];

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

const dataQualityIssues = computed((): DataQualityIssue[] => {
  const collectedIssues: DataQualityIssue[] = [];
  if (rawData.value.length === 0) return collectedIssues;

  const { dateColumn, geoColumn, metricColumns } = currentDataSource.value.columns;
  let hasUnfixableFormatErrorFlag = false;
  const encounteredGenericIssueMessages = new Set<string>();

  // --- Phase 1: Identify All Potential Issues (Missing, Malformed) ---
  for (let i = 0; i < rawData.value.length; i++) {
    const row = rawData.value[i];
    const rowIndexStr = `Row ${i + 1}`;

    // Date Column
    if (dateColumn) {
      const dateVal = row[dateColumn];
      if (dateVal === undefined || dateVal === null || String(dateVal).trim() === '') {
        const msg = `Missing values in Date Column '${dateColumn}'. Fixable by removing rows.`;
        if (!encounteredGenericIssueMessages.has(msg)) {
          collectedIssues.push({ type: DataQualityIssueType.MissingDate, message: msg });
          encounteredGenericIssueMessages.add(msg);
        }
      } else {
        const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
        if (
          !dateRegex.test(String(dateVal)) ||
          isNaN(new Date(dateVal as string | number | Date).getTime())
        ) {
          collectedIssues.push({
            type: DataQualityIssueType.InvalidDateFormat,
            message: `${rowIndexStr}: Invalid date format in '${dateColumn}' column: '${String(
              dateVal,
            )}'. Expected YYYY-MM-DD.`,
            isUnfixableFormatError: true,
          });
          hasUnfixableFormatErrorFlag = true;
        }
      }
    }

    // Metric Columns
    if (metricColumns.length > 0) {
      for (const metric of metricColumns) {
        const metricVal = row[metric];
        if (metricVal === undefined || metricVal === null) {
          const msg = `Missing values in Metric Column '${metric}'. Fixable by filling with 0.`;
          if (!encounteredGenericIssueMessages.has(msg)) {
            collectedIssues.push({ type: DataQualityIssueType.MissingMetric, message: msg });
            encounteredGenericIssueMessages.add(msg);
          }
        } else if (typeof metricVal !== 'number') {
          collectedIssues.push({
            type: DataQualityIssueType.NonNumericMetric,
            message: `${rowIndexStr}: Non-numeric value in Metric column '${metric}': '${metricVal}'.`,
            isUnfixableFormatError: true,
          });
          hasUnfixableFormatErrorFlag = true;
        }
      }
    }

    // Geo Column
    if (geoColumn) {
      const geoVal = row[geoColumn];
      if (geoVal === undefined || geoVal === null || String(geoVal).trim() === '') {
        const msg = `Missing values in Geo Column '${geoColumn}'. Fixable by removing rows.`;
        if (!encounteredGenericIssueMessages.has(msg)) {
          collectedIssues.push({ type: DataQualityIssueType.MissingGeo, message: msg });
          encounteredGenericIssueMessages.add(msg);
        }
      }
    }
  }

  if (hasUnfixableFormatErrorFlag) {
    const displayIssuesWhenCritical: DataQualityIssue[] = [];
    displayIssuesWhenCritical.push({
      type: DataQualityIssueType.InvalidDateFormat,
      message:
        'Critical data format errors found (e.g., unparsable dates, non-numeric metrics). These must be corrected. Other checks are paused.',
      isUnfixableFormatError: true,
    });

    collectedIssues.forEach((issue) => {
      if (issue.isUnfixableFormatError && issue.message !== displayIssuesWhenCritical[0].message) {
        displayIssuesWhenCritical.push(issue);
      }
    });

    if (
      dateColumn &&
      encounteredGenericIssueMessages.has(
        `Missing values in Date Column '${dateColumn}'. Fixable by removing rows.`,
      )
    ) {
      displayIssuesWhenCritical.push({
        type: DataQualityIssueType.MissingDate,
        message: `Missing values in Date Column '${dateColumn}'. Fixable by removing rows.`,
      });
    }
    if (
      geoColumn &&
      encounteredGenericIssueMessages.has(
        `Missing values in Geo Column '${geoColumn}'. Fixable by removing rows.`,
      )
    ) {
      displayIssuesWhenCritical.push({
        type: DataQualityIssueType.MissingGeo,
        message: `Missing values in Geo Column '${geoColumn}'. Fixable by removing rows.`,
      });
    }
    metricColumns.forEach((metric) => {
      if (
        encounteredGenericIssueMessages.has(
          `Missing values in Metric Column '${metric}'. Fixable by filling with 0.`,
        )
      ) {
        displayIssuesWhenCritical.push({
          type: DataQualityIssueType.MissingMetric,
          message: `Missing values in Metric Column '${metric}'. Fixable by filling with 0.`,
        });
      }
    });

    const finalDisplayMessages = new Set<string>();
    return displayIssuesWhenCritical.filter((issue) => {
      if (finalDisplayMessages.has(issue.message)) return false;
      finalDisplayMessages.add(issue.message);
      return true;
    });
  }

  // --- Phase 2: Integrity Checks (Date Gaps, Duplicates) ---
  const phase2IntegrityIssues: DataQualityIssue[] = [];

  // Date Gaps
  if (dateColumn && rawData.value.length > 1) {
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    const validDateRows = rawData.value.filter((row) => {
      const dateVal = row[dateColumn];
      return (
        dateVal !== undefined &&
        dateVal !== null &&
        String(dateVal).trim() !== '' &&
        dateRegex.test(String(dateVal)) &&
        !isNaN(new Date(dateVal as string | number | Date).getTime())
      );
    });

    if (validDateRows.length > 1) {
      const dataSorted = [...validDateRows].sort((a, b) => {
        if (geoColumn && a[geoColumn] !== b[geoColumn]) {
          if (a[geoColumn] == null && b[geoColumn] != null) return 1;
          if (a[geoColumn] != null && b[geoColumn] == null) return -1;
          if (a[geoColumn] == null && b[geoColumn] == null) return 0;
          return String(a[geoColumn]).localeCompare(String(b[geoColumn]));
        }
        return (
          new Date(a[dateColumn] as string | number | Date).getTime() -
          new Date(b[dateColumn] as string | number | Date).getTime()
        );
      });

      let lastDateObj: Date | null = null;
      let lastGeoVal: string | null = null;
      const dateGapMessages = new Set<string>();
      for (const currentRow of dataSorted) {
        const currentDateObj = new Date(currentRow[dateColumn] as string | number | Date);
        const currentGeoVal = geoColumn
          ? String(currentRow[geoColumn] ?? '__NULL_GEO__')
          : '__NO_GEO__';
        if (lastDateObj && (geoColumn ? currentGeoVal === lastGeoVal : true)) {
          const diffTime = currentDateObj.getTime() - lastDateObj.getTime();
          const diffDays = Math.round(diffTime / (1000 * 60 * 60 * 24));
          if (diffDays > 1) {
            dateGapMessages.add(
              `Date gap detected${geoColumn && lastGeoVal !== '__NULL_GEO__' ? ` for Geo: '${lastGeoVal}'` : ''} between ${lastDateObj.toISOString().split('T')[0]} and ${currentDateObj.toISOString().split('T')[0]} (${diffDays - 1} missing day(s)).`,
            );
          }
        }
        lastDateObj = currentDateObj;
        lastGeoVal = currentGeoVal;
      }
      dateGapMessages.forEach((msg) =>
        phase2IntegrityIssues.push({ type: DataQualityIssueType.DateGap, message: msg }),
      );
    }
  }

  // Duplicate Geo/Date
  if (dateColumn && geoColumn && rawData.value.length > 0) {
    const seenCombinations = new Set<string>();
    const duplicateRowMessages = new Set<string>();
    const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
    const validRowsForDuplicateCheck = rawData.value.filter((row) => {
      const dateVal = row[dateColumn];
      const geoVal = row[geoColumn];
      return (
        dateVal !== undefined &&
        dateVal !== null &&
        String(dateVal).trim() !== '' &&
        geoVal !== undefined &&
        geoVal !== null &&
        String(geoVal).trim() !== '' &&
        dateRegex.test(String(dateVal)) &&
        !isNaN(new Date(dateVal as string | number | Date).getTime())
      );
    });

    for (const row of validRowsForDuplicateCheck) {
      const dateVal = row[dateColumn];
      const geoVal = row[geoColumn];
      const formattedDate = new Date(dateVal as string | number | Date).toISOString().split('T')[0];
      const sGeoVal = String(geoVal);
      const combination = `${sGeoVal}|${formattedDate}`;
      if (seenCombinations.has(combination)) {
        duplicateRowMessages.add(
          `Duplicate entry found for Geo: '${sGeoVal}', Date: '${formattedDate}'.`,
        );
      } else {
        seenCombinations.add(combination);
      }
    }
    duplicateRowMessages.forEach((msg) =>
      phase2IntegrityIssues.push({ type: DataQualityIssueType.DuplicateGeoDate, message: msg }),
    );
  }

  const allDetectedIssues = [
    ...collectedIssues.filter((issue) => !issue.isUnfixableFormatError),
    ...phase2IntegrityIssues,
  ];

  const finalUniqueMessages = new Set<string>();
  return allDetectedIssues.filter((issue) => {
    if (finalUniqueMessages.has(issue.message)) {
      return false;
    }
    finalUniqueMessages.add(issue.message);
    return true;
  });
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

  rawData.value = data.map((row, index) => ({
    ...row,
    __index: index,
  }));

  const firstRow = data[0];
  columns.value = Object.keys(firstRow)
    .filter((key) => key !== '__index')
    .map((key) => ({
      name: key,
      label: key,
      field: key,
    }));

  columnRoles.value = {};
  columns.value.forEach((col) => {
    columnRoles.value[col.name] = ColumnRole.Ignore;
  });

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
          columns: true,
          skip_empty_lines: true,
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

    processData(parsedData);

    // Set data source name from file if not already set
    if (!currentDataSource.value.name && file.value.name) {
      currentDataSource.value.name = file.value.name.replace(/\.[^/.]+$/, '');
    }
    currentDataSource.value.sourceLink = file.value.name;

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

    processData(res.data as Record<string, unknown>[]);

    // Auto-set data source name from URL if not set
    if (!currentDataSource.value.name) {
      const url = new URL(externalSourceUrl.value);
      currentDataSource.value.name = url.pathname.split('/').pop() || 'External Data Source';
    }

    $q.notify({
      type: 'positive',
      message: 'Successfully loaded data from external source',
    });
  } catch (error) {
    assertIsError(error);
    showError('Failed to load external data: ' + error.message);
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
    // Always update from rawData if it has content, for both new and existing data sources
    if (rawData.value.length > 0) {
      currentDataSource.value.data = dataSourcesStore.normalizeRawData(
        rawData.value,
        currentDataSource.value.columns,
      );
    } else if (isEditing.value && !rawData.value.length) {
      currentDataSource.value.data = dataSourcesStore.normalizeRawData(
        [],
        currentDataSource.value.columns,
      );
    }

    const res = await dataSourcesStore.saveDataSource(currentDataSource.value);
    if (!res) return;

    emit('saved', res || currentDataSource.value);
  } catch (error) {
    showError('Failed to save data source: ' + error);
  } finally {
    saving.value = false;
  }
}

function onCancel() {
  emit('canceled');
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleColumnRoleChange(columnName: string, newRole: any) {
  const roleValue = typeof newRole === 'object' && newRole !== null ? newRole.value : newRole;

  if (columnName) {
    columnRoles.value[columnName] = roleValue;
  }

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

function openFixDialog() {
  fixSelection.value = {
    removeMissingCritical: availableFixTypes.value.removeMissingCritical,
    fillMissingMetrics: availableFixTypes.value.fillMissingMetrics,
    fillDateGaps: availableFixTypes.value.fillDateGaps,
    deduplicateRows: availableFixTypes.value.deduplicateRows,
  };
  showFixSelectionDialog.value = true;
}

function applySelectedFixes() {
  showFixSelectionDialog.value = false;
  executeFixes(fixSelection.value);
}

function executeFixes(selection: FixSelection) {
  if (!rawData.value.length) return;

  isFixingIssues.value = true;
  let rowsRemovedCount = 0;
  let metricsFilledCount = 0;
  let dateGapsFilledCount = 0;
  let deduplicatedRowsCount = 0;

  const { geoColumn, dateColumn, metricColumns } = currentDataSource.value.columns;

  let currentProcessedData: Record<string, unknown>[] = [...rawData.value];

  // 1. Remove rows with missing geo/date values
  if (selection.removeMissingCritical) {
    const tempData = currentProcessedData
      .map((row) => {
        let rowIsValid = true;
        if (geoColumn && !row[geoColumn]) {
          rowIsValid = false;
        }
        if (dateColumn && !row[dateColumn]) {
          rowIsValid = false;
        }
        return rowIsValid ? row : null;
      })
      .filter((row): row is Record<string, unknown> => row !== null);
    rowsRemovedCount = currentProcessedData.length - tempData.length;
    currentProcessedData = tempData;
  }

  // 2. Fill missing metric values
  if (selection.fillMissingMetrics) {
    currentProcessedData = currentProcessedData.map((row) => {
      const newRow = { ...row };
      if (metricColumns.length > 0) {
        metricColumns.forEach((metric) => {
          if (!newRow[metric] && newRow[metric] !== 0) {
            newRow[metric] = 0;
            metricsFilledCount++;
          }
        });
      }
      return newRow;
    });
  }

  // 3. Fill date gaps
  if (selection.fillDateGaps && dateColumn && currentProcessedData.length > 0) {
    const newRowsToAdd: Record<string, unknown>[] = [];
    const dataByGeo: Record<string, Record<string, unknown>[]> = {};

    if (geoColumn) {
      currentProcessedData.forEach((row) => {
        const geoVal = row[geoColumn];
        let geo: string;
        if (geoVal === null || geoVal === undefined) {
          geo = '__NULL_GEO__';
        } else if (
          typeof geoVal === 'string' ||
          typeof geoVal === 'number' ||
          typeof geoVal === 'boolean'
        ) {
          geo = String(geoVal);
        } else {
          geo = JSON.stringify(geoVal);
        }
        if (!dataByGeo[geo]) dataByGeo[geo] = [];
        dataByGeo[geo].push(row);
      });
    } else {
      dataByGeo['__NO_GEO__'] = [...currentProcessedData];
    }

    Object.keys(dataByGeo).forEach((geoKey) => {
      const geoGroupData = dataByGeo[geoKey].sort((a, b) => {
        const dateA = a[dateColumn];
        const dateB = b[dateColumn];
        const timeA =
          typeof dateA === 'string' || typeof dateA === 'number' || dateA instanceof Date
            ? new Date(dateA).getTime()
            : 0;
        const timeB =
          typeof dateB === 'string' || typeof dateB === 'number' || dateB instanceof Date
            ? new Date(dateB).getTime()
            : 0;
        return timeA - timeB;
      });
      if (geoGroupData.length === 0) return;

      const firstDateVal = geoGroupData[0][dateColumn];
      const lastDateVal = geoGroupData[geoGroupData.length - 1][dateColumn];

      if (
        !(
          typeof firstDateVal === 'string' ||
          typeof firstDateVal === 'number' ||
          firstDateVal instanceof Date
        ) ||
        !(
          typeof lastDateVal === 'string' ||
          typeof lastDateVal === 'number' ||
          lastDateVal instanceof Date
        )
      ) {
        return;
      }

      const minDate = new Date(firstDateVal);
      const maxDate = new Date(lastDateVal);
      const existingDates = new Set(
        geoGroupData
          .map((row) => {
            const dateVal = row[dateColumn];
            return typeof dateVal === 'string' ||
              typeof dateVal === 'number' ||
              dateVal instanceof Date
              ? new Date(dateVal).toISOString().split('T')[0]
              : '';
          })
          .filter((dateStr) => dateStr !== ''),
      );

      for (let d = new Date(minDate); d <= maxDate; d.setDate(d.getDate() + 1)) {
        const currentDateStr = d.toISOString().split('T')[0];
        if (!existingDates.has(currentDateStr)) {
          const newRow: Record<string, unknown> = { __index: -1 };
          if (geoColumn && geoKey !== '__NO_GEO__' && geoKey !== '__NULL_GEO__') {
            newRow[geoColumn] = geoKey;
          } else if (geoColumn) {
            newRow[geoColumn] = null;
          }
          newRow[dateColumn] = currentDateStr;
          metricColumns.forEach((metric) => {
            newRow[metric] = 0;
          });

          columns.value.forEach((colDef) => {
            if (!(colDef.name in newRow) && colDef.name !== '__index') {
              newRow[colDef.name] = undefined;
            }
          });

          newRowsToAdd.push(newRow);
          dateGapsFilledCount++;
        }
      }
    });

    if (newRowsToAdd.length > 0) {
      currentProcessedData.push(...newRowsToAdd);
      currentProcessedData.sort((a, b) => {
        if (geoColumn && a[geoColumn] !== b[geoColumn]) {
          const geoValA = a[geoColumn];
          const geoValB = b[geoColumn];
          let geoA_str: string;
          let geoB_str: string;

          if (geoValA === null || geoValA === undefined) geoA_str = '';
          else if (
            typeof geoValA === 'string' ||
            typeof geoValA === 'number' ||
            typeof geoValA === 'boolean'
          )
            geoA_str = String(geoValA);
          else geoA_str = JSON.stringify(geoValA);

          if (geoValB === null || geoValB === undefined) geoB_str = '';
          else if (
            typeof geoValB === 'string' ||
            typeof geoValB === 'number' ||
            typeof geoValB === 'boolean'
          )
            geoB_str = String(geoValB);
          else geoB_str = JSON.stringify(geoValB);

          return geoA_str.localeCompare(geoB_str);
        }
        const dateA = a[dateColumn];
        const dateB = b[dateColumn];
        const timeA =
          typeof dateA === 'string' || typeof dateA === 'number' || dateA instanceof Date
            ? new Date(dateA).getTime()
            : 0;
        const timeB =
          typeof dateB === 'string' || typeof dateB === 'number' || dateB instanceof Date
            ? new Date(dateB).getTime()
            : 0;
        return timeA - timeB;
      });
    }
  }

  // 4. Remove duplicate Geo/Date rows
  if (selection.deduplicateRows && dateColumn && geoColumn && currentProcessedData.length > 0) {
    const seenCombinations = new Set<string>();
    const uniqueRows: Record<string, unknown>[] = [];
    const originalLength = currentProcessedData.length;

    for (const row of currentProcessedData) {
      const dateVal = row[dateColumn];
      const geoVal = row[geoColumn];

      const formattedDate: string = new Date(dateVal as string | number | Date)
        .toISOString()
        .split('T')[0];

      let sGeoVal: string;
      if (geoVal === null || geoVal === undefined) {
        sGeoVal = 'N/A';
      } else if (
        typeof geoVal === 'string' ||
        typeof geoVal === 'number' ||
        typeof geoVal === 'boolean'
      ) {
        sGeoVal = String(geoVal);
      } else {
        sGeoVal = JSON.stringify(geoVal);
      }
      const combination = `${sGeoVal}|${formattedDate}`;

      if (!seenCombinations.has(combination)) {
        seenCombinations.add(combination);
        uniqueRows.push(row);
      }
    }
    deduplicatedRowsCount = originalLength - uniqueRows.length;
    currentProcessedData = uniqueRows;
  }

  rawData.value = currentProcessedData.map((row, index) => ({
    ...row,
    __index: index,
  }));

  let message = 'Fixes applied: ';
  const fixesApplied = [];
  if (rowsRemovedCount > 0)
    fixesApplied.push(`${rowsRemovedCount} row(s) with missing critical data removed`);
  if (metricsFilledCount > 0)
    fixesApplied.push(`${metricsFilledCount} missing metric value(s) filled`);
  if (dateGapsFilledCount > 0) fixesApplied.push(`${dateGapsFilledCount} date gap(s) filled`);
  if (deduplicatedRowsCount > 0)
    fixesApplied.push(`${deduplicatedRowsCount} duplicate row(s) removed`);

  if (fixesApplied.length === 0) {
    message = 'No fixes were applied based on selection or no relevant issues found.';
  } else {
    message += fixesApplied.join(', ') + '.';
  }

  $q.notify({
    type: fixesApplied.length > 0 ? 'positive' : 'info',
    message: message.trim(),
    position: 'top',
    timeout: 5000,
    actions: [{ icon: 'close', color: 'white', round: true, dense: true }],
  });

  isFixingIssues.value = false;
}

async function triggerValidation() {
  validateDataLoading.value = true;
  await new Promise((resolve) => setTimeout(resolve, 200));
  const issuesCount = dataQualityIssues.value.length;

  if (issuesCount > 0) {
    $q.notify({
      type: 'warning',
      message: `Validation complete. ${issuesCount} issue(s) found. Please review the Data Quality Issues section.`,
      position: 'top',
      timeout: 3000,
    });
  } else {
    $q.notify({
      type: 'positive',
      message: 'Validation complete. No data quality issues detected.',
      position: 'top',
      timeout: 3000,
    });
  }
  validateDataLoading.value = false;
}
</script>
