<template>
  <q-page class="q-pa-md">
    <div class="q-mb-lg flex justify-between items-center">
      <div>
        <h1 class="text-h4 q-mb-sm">Data Sources</h1>
        <p class="text-body1">Manage data sources for your geo-based marketing tests.</p>
      </div>
      <div class="row q-gutter-sm">
        <q-btn
          v-if="!isSynced"
          color="warning"
          icon="cloud_upload"
          label="Save to Server"
          @click="showSaveToServerDialog = true"
        />
        <q-btn color="primary" icon="add" label="New Data Source" @click="openDataSourceEditor()" />
      </div>
    </div>

    <!-- List of data sources -->
    <q-card v-if="datasources && datasources.length > 0">
      <q-card-section>
        <q-table
          :rows="datasources"
          :columns="columns"
          row-key="id"
          flat
          :pagination="{ rowsPerPage: 10 }"
          :loading="loading"
          :filter="search"
        >
          <template v-slot:top>
            <q-input
              v-model="search"
              debounce="300"
              placeholder="Search data sources"
              dense
              filled
              class="q-mb-md"
              style="max-width: 300px"
            >
              <template v-slot:append>
                <q-icon name="search" />
              </template>
            </q-input>
          </template>

          <template v-slot:body-cell-type="props">
            <q-td :props="props">
              <q-badge
                :color="props.row.sourceType === DataSourceType.Internal ? 'primary' : 'secondary'"
              >
                {{ props.row.sourceType === DataSourceType.Internal ? 'Internal' : 'External' }}
              </q-badge>
            </q-td>
          </template>

          <template v-slot:body-cell-status="props">
            <q-td :props="props">
              <q-badge :color="getDataSourceStatus(props.row).color">
                {{ getDataSourceStatus(props.row).label }}
              </q-badge>
            </q-td>
          </template>

          <template v-slot:body-cell-actions="props">
            <q-td :props="props" class="q-gutter-x-sm">
              <q-btn
                flat
                round
                size="sm"
                color="primary"
                icon="edit"
                @click="openDataSourceEditor(props.row.id)"
              >
                <q-tooltip>Edit</q-tooltip>
              </q-btn>

              <q-btn
                flat
                round
                size="sm"
                color="negative"
                icon="delete"
                @click="confirmDelete(props.row)"
              >
                <q-tooltip>Delete</q-tooltip>
              </q-btn>

              <q-btn
                flat
                round
                size="sm"
                color="info"
                icon="visibility"
                @click="viewData(props.row.id)"
              >
                <q-tooltip>View Data</q-tooltip>
              </q-btn>
            </q-td>
          </template>
        </q-table>
      </q-card-section>
    </q-card>

    <!-- Empty state -->
    <q-card v-else class="text-center q-pa-lg">
      <q-icon name="source" size="6rem" color="grey-5" />
      <div class="text-h6 q-mt-md">No Data Sources Yet</div>
      <p class="q-mb-md">Create your first data source to start designing geo tests</p>
      <q-btn color="primary" label="Create Data Source" @click="openDataSourceEditor()" />
    </q-card>

    <!-- DataSource Editor Dialog -->
    <q-dialog v-model="showEditor" persistent maximized>
      <q-card>
        <q-card-section class="row items-center q-pb-none">
          <div class="text-h6">
            {{ editingId ? 'Edit Data Source' : 'New Data Source' }}
          </div>
          <q-space />
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>

        <q-separator class="q-my-sm" />

        <q-card-section class="q-pa-none">
          <DataSourceEditor
            :editing-id="editingId"
            @saved="onDataSourceSaved"
            @canceled="showEditor = false"
          />
        </q-card-section>
      </q-card>
    </q-dialog>

    <!-- Save to Server Dialog -->
    <q-dialog v-model="showSaveToServerDialog">
      <q-card>
        <q-card-section class="row items-center">
          <q-avatar icon="cloud_upload" color="primary" text-color="white" />
          <span class="q-ml-sm"> Save changes to server? </span>
        </q-card-section>

        <q-card-section>
          <p>You have made changes to your data sources that need to be saved to the server.</p>
        </q-card-section>

        <q-card-actions align="right">
          <q-btn flat label="Cancel" color="grey" v-close-popup />
          <q-btn
            flat
            label="Save to Server"
            color="primary"
            @click="saveToServer"
            :loading="saving"
            v-close-popup
          />
        </q-card-actions>
      </q-card>
    </q-dialog>

    <!-- Delete Confirmation Dialog -->
    <q-dialog v-model="showDeleteConfirm" persistent>
      <q-card>
        <q-card-section class="row items-center">
          <q-avatar icon="delete" color="negative" text-color="white" />
          <span class="q-ml-sm">
            Are you sure you want to delete this data source? This action cannot be undone.
          </span>
        </q-card-section>

        <q-card-actions align="right">
          <q-btn flat label="Cancel" color="primary" v-close-popup />
          <q-btn flat label="Delete" color="negative" @click="deleteDataSource" v-close-popup />
        </q-card-actions>
      </q-card>
    </q-dialog>

    <!-- View Data Dialog -->
    <q-dialog v-model="showDataView" maximized>
      <q-card>
        <q-card-section class="row items-center">
          <div class="text-h6">Data Preview: {{ currentDataSource?.name }}</div>
          <q-space />
          <q-btn icon="close" flat round dense v-close-popup />
        </q-card-section>

        <q-separator class="q-my-sm" />

        <q-card-section>
          <template v-if="currentDataSource?.data">
            <div class="row q-mb-md">
              <div class="col-12 col-md-3">
                <q-card class="bg-primary text-white q-pa-sm">
                  <q-card-section>
                    <div class="text-h6">Geo Units</div>
                    <div class="text-h4">
                      {{ currentDataSource.data.geo.length }}
                    </div>
                  </q-card-section>
                </q-card>
              </div>
              <div class="col-12 col-md-3">
                <q-card class="bg-secondary text-white q-pa-sm">
                  <q-card-section>
                    <div class="text-h6">Time Points</div>
                    <div class="text-h4">
                      {{ currentDataSource.data.numberOfDays }}
                    </div>
                  </q-card-section>
                </q-card>
              </div>
              <div class="col-12 col-md-3">
                <q-card class="bg-positive text-white q-pa-sm">
                  <q-card-section>
                    <div class="text-h6">Total Rows</div>
                    <div class="text-h4">
                      {{ getTotalRows(currentDataSource.data) }}
                    </div>
                  </q-card-section>
                </q-card>
              </div>
            </div>

            <!-- Geo selector -->
            <div class="q-mb-md">
              <q-select
                v-model="selectedGeo"
                :options="currentDataSource.data.geo"
                label="Select Geo Unit"
                filled
                emit-value
                map-options
                @update:model-value="updateDataTable"
              />
            </div>

            <!-- Data table -->
            <q-table
              v-if="selectedGeo && dataTable.length > 0"
              :rows="dataTable"
              :columns="dataColumns"
              row-key="date"
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

            <div v-else class="text-center q-pa-md">
              <p>Select a geo unit to view data</p>
            </div>
          </template>

          <div v-else-if="currentDataSource?.sourceType === DataSourceType.External">
            <div class="text-center q-pa-lg">
              <q-spinner color="primary" size="3em" class="q-mb-md" v-if="loadingData" />
              <div v-else>
                <q-icon name="cloud_download" size="3em" color="grey-7" />
                <div class="text-h6 q-mt-md">External Data Not Loaded</div>
                <p class="q-mb-md">
                  This data source is connected to an external source but data has not been loaded
                  yet.
                </p>
                <q-btn
                  color="primary"
                  label="Load Data"
                  @click="loadExternalData(currentDataSource.id)"
                  :loading="loadingData"
                />
              </div>
            </div>
          </div>

          <div v-else class="text-center q-pa-md">
            <p>No data available for this data source</p>
          </div>
        </q-card-section>
      </q-card>
    </q-dialog>
  </q-page>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, toRefs, watch } from 'vue';
import type { QTableColumn } from 'quasar';
import { useQuasar } from 'quasar';
import DataSourceEditor from 'components/DataSourceEditor.vue';
import type { DataSource, DataRow } from 'stores/datasources';
import { useDataSourcesStore, DataSourceType } from 'stores/datasources';
import { assertIsError } from 'src/helpers/utils';

// Quasar utilities
const $q = useQuasar();

// Store
const dataSourcesStore = useDataSourcesStore();
// Use toRefs to make the store properties reactive
const { datasources, loading: storeLoading, isSynced } = toRefs(dataSourcesStore);

// UI state
const loading = computed(() => storeLoading.value);
const saving = ref(false);
const search = ref('');
const showEditor = ref(false);
const editingId = ref<string | null>(null);
const showDeleteConfirm = ref(false);
const selectedDataSourceToDelete = ref<DataSource | null>(null);
const showDataView = ref(false);
const currentDataSource = ref<DataSource | null>(null);
const selectedGeo = ref<string | null>(null);
const dataTable = ref<DataRow[]>([]);
const dataSearch = ref('');
const loadingData = ref(false);
const showSaveToServerDialog = ref(false);

// Watch for changes in sync status
watch(isSynced, (newVal) => {
  if (!newVal) {
    // Show notification about unsaved changes
    $q.notify({
      type: 'warning',
      message: 'You have unsaved changes',
      actions: [
        {
          label: 'Save',
          color: 'white',
          handler: () => {
            showSaveToServerDialog.value = true;
          },
        },
      ],
      timeout: 5000,
    });
  }
});

// Table columns
const columns = [
  {
    name: 'name',
    label: 'Name',
    field: 'name',
    sortable: true,
    align: 'left',
  },
  {
    name: 'type',
    label: 'Type',
    field: 'sourceType',
    sortable: true,
    align: 'left',
  },
  {
    name: 'geoColumn',
    label: 'Geo Column',
    field: 'geoColumn',
    sortable: true,
    align: 'left',
  },
  {
    name: 'dateColumn',
    label: 'Date Column',
    field: 'dateColumn',
    sortable: true,
    align: 'left',
  },
  {
    name: 'metricColumns',
    label: 'Metrics',
    field: (row) => {
      return Array.isArray(row.metricColumns) ? row.metricColumns.join(', ') : row.metricColumns;
    },
    sortable: false,
    align: 'left',
  },
  {
    name: 'status',
    label: 'Status',
    field: 'id',
    sortable: false,
    align: 'center',
  },
  {
    name: 'actions',
    label: 'Actions',
    field: 'actions',
    sortable: false,
    align: 'center',
  },
] as QTableColumn[];

// Data preview columns
const dataColumns = computed(() => {
  return [
    {
      name: 'date',
      label: 'Date',
      field: 'date',
      sortable: true,
      align: 'left',
    },
    {
      name: 'metric',
      label: currentDataSource.value?.metricColumns?.[0] || 'Metric',
      field: 'metric',
      sortable: true,
      align: 'right',
    },
    {
      name: 'cost',
      label: 'Cost',
      field: 'cost',
      sortable: true,
      align: 'right',
    },
  ] as QTableColumn[];
});

// Methods
onMounted(async () => {
  try {
    // Load data sources from the store
    await dataSourcesStore.loadDataSources();
  } catch (error) {
    assertIsError(error);
    $q.notify({
      type: 'negative',
      message: 'Failed to load data sources: ' + error.message,
      position: 'top',
    });
  }
});

function openDataSourceEditor(id?: string) {
  editingId.value = id || null;
  showEditor.value = true;
}

function onDataSourceSaved() {
  showEditor.value = false;

  $q.notify({
    type: 'positive',
    message: editingId.value
      ? 'Data source updated successfully'
      : 'New data source created successfully',
    position: 'top',
    actions: [
      {
        label: 'Save to Server',
        color: 'white',
        handler: () => {
          showSaveToServerDialog.value = true;
        },
      },
    ],
  });

  // Reset editing state
  editingId.value = null;
}

async function saveToServer() {
  saving.value = true;
  try {
    await dataSourcesStore.saveToServer();

    $q.notify({
      type: 'positive',
      message: 'Changes saved to server successfully',
      position: 'top',
    });
  } catch (error) {
    console.error('Error saving to server:', error);
    $q.notify({
      type: 'negative',
      message: 'Failed to save changes to server',
      position: 'top',
    });
  } finally {
    saving.value = false;
  }
}

function confirmDelete(dataSource: DataSource) {
  selectedDataSourceToDelete.value = dataSource;
  showDeleteConfirm.value = true;
}

async function deleteDataSource() {
  if (!selectedDataSourceToDelete.value) return;

  try {
    // Delete from the store
    await dataSourcesStore.deleteDataSource(selectedDataSourceToDelete.value.id);

    $q.notify({
      type: 'positive',
      message: 'Data source deleted successfully',
      position: 'top',
    });
  } catch (error) {
    console.error('Error deleting data source:', error);
    $q.notify({
      type: 'negative',
      message: 'Failed to delete data source',
      position: 'top',
    });
  } finally {
    selectedDataSourceToDelete.value = null;
  }
}

function viewData(id: string) {
  const dataSource = datasources.value.find((d) => d.id === id);
  if (!dataSource) return;

  currentDataSource.value = dataSource;

  // If data is available and there are geo units, initialize the view
  if (dataSource.data && dataSource.data.geo.length > 0) {
    selectedGeo.value = dataSource.data.geo[0];
    updateDataTable();
  }

  showDataView.value = true;
}

async function loadExternalData(id: string) {
  loadingData.value = true;
  try {
    // Load data for external source
    const updatedSource = await dataSourcesStore.loadDataSourceData(id);

    // Update the current data source reference
    if (updatedSource) {
      currentDataSource.value = updatedSource;

      // If data is now available, initialize the view
      if (updatedSource.data && updatedSource.data.geo.length > 0) {
        selectedGeo.value = updatedSource.data.geo[0];
        updateDataTable();
      }
    }
  } catch (error) {
    console.error('Error loading external data:', error);
    $q.notify({
      type: 'negative',
      message: 'Failed to load external data',
      position: 'top',
    });
  } finally {
    loadingData.value = false;
  }
}

function updateDataTable() {
  if (!currentDataSource.value?.data || !selectedGeo.value) {
    dataTable.value = [];
    return;
  }

  // Get data for selected geo
  const geoData = currentDataSource.value.data.rows[selectedGeo.value] || [];
  dataTable.value = [...geoData];
}

function getDataSourceStatus(dataSource: DataSource) {
  if (!dataSource.data) {
    return { color: 'warning', label: 'No Data' };
  }

  const rowCount = Object.values(dataSource.data.rows).reduce(
    (acc: number, rows) => acc + rows.length,
    0,
  );

  if (rowCount === 0) {
    return { color: 'warning', label: 'Empty' };
  }

  return { color: 'positive', label: 'Active' };
}

function getTotalRows(data: DataSource['data']) {
  if (!data) return 0;

  return Object.values(data.rows).reduce((acc, rows) => acc + rows.length, 0);
}
</script>
