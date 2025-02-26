<template>
  <q-layout view="hHh Lpr lFf">
    <q-header elevated>
      <q-toolbar>
        <q-btn
          flat
          dense
          round
          icon="menu"
          aria-label="Menu"
          @click="toggleLeftDrawer"
        />

        <q-toolbar-title>
          <router-link to="/" style="text-decoration: none; color: white"
            >GeoFlex Testing Platform</router-link
          ></q-toolbar-title
        >
      </q-toolbar>
    </q-header>

    <q-drawer v-model="leftDrawerOpen" show-if-above bordered>
      <SideMenu />
    </q-drawer>

    <q-page-container>
      <router-view />
    </q-page-container>

    <div class="text-body1 text-center q-ma-sm">
      &copy;&nbsp;Google gTech Ads, 2025.
    </div>
  </q-layout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import SideMenu from 'components/SideMenu.vue';

defineOptions({
  name: 'MainLayout',
});

const leftDrawerOpen = ref(false);

function toggleLeftDrawer() {
  leftDrawerOpen.value = !leftDrawerOpen.value;
}
</script>
