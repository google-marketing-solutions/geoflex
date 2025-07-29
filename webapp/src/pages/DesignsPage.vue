<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="row items-center justify-between q-mb-md">
        <div class="text-h4">Saved Designs</div>
        <q-btn icon="refresh" round flat @click="fetchDesigns" />
      </div>
      <design-list
        :designs="designs"
        :showAnalyze="true"
        :show-meta="true"
        :show-delete="true"
        @analyze="analyzeDesign"
        @delete="deleteDesign"
      />
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { getApiUi, deleteApiUi } from 'src/boot/axios';
import DesignList from 'src/components/DesignList.vue';
import type { SavedDesign } from 'src/components/models';
import { useRouter } from 'vue-router';

const designs = ref<SavedDesign[]>([]);
const router = useRouter();

async function fetchDesigns() {
  const response = await getApiUi<SavedDesign[]>('designs', {}, 'Loading designs');
  if (response && response.data) {
    designs.value = response.data;
  }
}

async function deleteDesign(design: SavedDesign) {
  const response = await deleteApiUi(
    `designs/${design.design.design_id}`,
    'Deleting design...',
    `Are you sure you want to delete design ${design.design.design_id}?`,
  );
  if (response) {
    await fetchDesigns();
  }
}

function analyzeDesign(design: SavedDesign) {
  router
    .push({
      name: 'analysis',
      params: {
        designId: design.design.design_id,
      },
    })
    .catch((err) => {
      console.error('Router push failed:', err);
    });
}

onMounted(async () => {
  await fetchDesigns();
});
</script>
