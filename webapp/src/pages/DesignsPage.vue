<template>
  <q-page padding>
    <div class="q-pa-md">
      <div class="row items-center justify-between q-mb-md">
        <div class="text-h4">Saved Designs</div>
        <q-btn icon="refresh" round flat @click="reloadDesigns" />
      </div>
      <design-list
        :designs="designsStore.designs"
        :showAnalyze="true"
        :show-meta="true"
        :show-delete="true"
        default-sort="recency"
        @analyze="analyzeDesign"
        @delete="deleteDesign"
        @update="updateDesign"
      />
    </div>
  </q-page>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import DesignList from 'src/components/DesignList.vue';
import type { SavedDesign } from 'src/components/models';
import { useRouter } from 'vue-router';
import { useDesignsStore } from 'src/stores/designs';

const router = useRouter();
const designsStore = useDesignsStore();

async function reloadDesigns() {
  await designsStore.loadDesigns(true);
}

async function deleteDesign(design: SavedDesign) {
  await designsStore.deleteDesign(design);
}

async function updateDesign(design: SavedDesign, newValues: Partial<SavedDesign>) {
  await designsStore.updateDesign(design, newValues);
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
  await designsStore.loadDesigns();
});
</script>
