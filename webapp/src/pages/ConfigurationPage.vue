<template>
  <q-page padding>
    <q-card class="configuration-card" >
      <q-card-section>
        <div class="text-h5 q-mb-md">Application Configuration</div>

        <q-form @submit="updateConfiguration" class="q-gutter-md" >
          <!-- Master Spreadsheet ID -->
          <q-input
            v-model="masterSpreadsheetId"
            label="Master Spreadsheet ID *"
            filled
            :rules="[(val) => !!val || 'Spreadsheet ID is required']"
            :loading="loading"
            :disable="loading"
          >
            <template v-slot:prepend>
              <q-icon name="table_chart" />
            </template>
            <template v-slot:append v-if="masterSpreadsheetId">
              <q-btn
                round
                flat
                dense
                icon="open_in_new"
                type="a"
                :href="`https://docs.google.com/spreadsheets/d/${masterSpreadsheetId}/edit`"
                target="_blank"
                class="q-ml-sm"
              >
                <q-tooltip>Open in Google Sheets</q-tooltip>
              </q-btn>
            </template>
            <template v-slot:hint>
              The ID of your master Google Spreadsheet where datasource definitions are stored
            </template>
          </q-input>

          <!-- Action Buttons -->
          <div class="row justify-between q-mt-lg">
            <q-btn
              label="Reload"
              color="grey-7"
              flat
              :disable="loading"
              @click="reload"
              icon="sync"
            />

            <div class="row q-gutter-sm">
              <q-btn
                label="Share With Me"
                color="secondary"
                icon="share"
                :loading="sharingLoading"
                :disable="loading || !masterSpreadsheetId"
                @click="shareSpreadsheet"
              />

              <q-btn
                label="Update"
                type="submit"
                color="primary"
                icon="save"
                :disable="!hasChanges()"
                :loading="loading"
              />

              <q-btn
                label="Recreate"
                color="warning"
                icon="create"
                :loading="loading"
                @click="createSpreadsheet"
              />
            </div>
          </div>
        </q-form>
      </q-card-section>
    </q-card>

    <!-- Spreadsheet Visibility & Information -->
    <q-card class="q-mt-md" v-if="masterSpreadsheetId">
      <q-card-section>
        <div class="text-h6">Spreadsheet Information</div>
        <p class="text-body1 q-mb-md">
          This is where your data source definitions are stored. Make sure you have access to this
          spreadsheet.
        </p>

        <q-list bordered separator>
          <q-item>
            <q-item-section avatar>
              <q-icon name="table_chart" color="primary" />
            </q-item-section>
            <q-item-section>
              <q-item-label>Master Spreadsheet</q-item-label>
              <q-item-label caption>
                <a
                  :href="`https://docs.google.com/spreadsheets/d/${masterSpreadsheetId}/edit`"
                  target="_blank"
                  class="text-primary"
                >
                  {{ masterSpreadsheetId }}
                </a>
              </q-item-label>
            </q-item-section>
          </q-item>
        </q-list>
      </q-card-section>
    </q-card>

    <!-- Instructions Card -->
    <q-card class="q-mt-md">
      <q-card-section>
        <div class="text-h6">Setup Instructions</div>

        <ol class="q-pl-md">
          <li class="q-mb-sm">
            Enter a Google Spreadsheet ID or create a new spreadsheet
            <a
              href="https://docs.google.com/spreadsheets/create"
              target="_blank"
              class="text-primary"
              >here</a
            >.
          </li>
          <li class="q-mb-sm">
            If you're using an existing spreadsheet, make sure it has a sheet named "DataSources"
            with the correct headers: id, name, created_at, updated_at, type, source_link, config.
          </li>
          <li class="q-mb-sm">
            After updating the configuration, click "Share With Me" to ensure you have access.
          </li>
          <li class="q-mb-sm">Go to the Data Sources page to start managing your data sources.</li>
        </ol>
      </q-card-section>
    </q-card>
  </q-page>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useQuasar } from 'quasar';
import axios from 'axios';

// Quasar utilities
const $q = useQuasar();

// Form state
const masterSpreadsheetId = ref('');
const originalId = ref('');
const loading = ref(false);
const sharingLoading = ref(false);

// Load current configuration
onMounted(async () => {
  await loadConfiguration();
});

async function loadConfiguration() {
  loading.value = true;

  try {
    const response = await axios.get('/api/config');
    masterSpreadsheetId.value = response.data.spreadsheet_id;
    originalId.value = response.data.spreadsheet_id;
  } catch (error) {
    showError('Failed to load configuration');
    console.error('Error loading configuration:', error);
  } finally {
    loading.value = false;
  }
}

function hasChanges() {
  return originalId.value !== masterSpreadsheetId.value;
}

// Update configuration
async function updateConfiguration() {
  if (!masterSpreadsheetId.value) return;

  loading.value = true;

  try {
    await axios.put('/api/config', {
      spreadsheet_id: masterSpreadsheetId.value,
    });

    originalId.value = masterSpreadsheetId.value;

    $q.notify({
      type: 'positive',
      message: 'Configuration updated successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to update configuration');
    console.error('Error updating configuration:', error);
  } finally {
    loading.value = false;
  }
}

// Share spreadsheet with current user
async function shareSpreadsheet() {
  if (!masterSpreadsheetId.value) return;

  sharingLoading.value = true;

  try {
    const response = await axios.post('/api/config/share');

    $q.notify({
      type: 'positive',
      message: response.data.message || 'Spreadsheet shared successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to share spreadsheet');
    console.error('Error sharing spreadsheet:', error);
  } finally {
    sharingLoading.value = false;
  }
}

async function createSpreadsheet() {
  loading.value = true;

  try {
    const response = await axios.post('/api/config/recreate');
    masterSpreadsheetId.value = response.data.spreadsheet_id;
    originalId.value = response.data.spreadsheet_id;

    $q.notify({
      type: 'positive',
      message: response.data.message || 'Spreadsheet recreated successfully',
      position: 'top',
    });
  } catch (error) {
    showError('Failed to create master spreadsheet');
    console.error('Error creating spreadsheet:', error);
  } finally {
    loading.value = false;
  }
}

// Reset form to original values
function reload() {
  loadConfiguration();
}

// Show error notification
function showError(message: string) {
  $q.notify({
    type: 'negative',
    message,
    position: 'top',
  });
}
</script>

<style scoped>
.configuration-card {
  max-width: 800px;
  margin: 0 auto;
}
</style>
