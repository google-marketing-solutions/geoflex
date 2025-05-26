<template>
  <q-page class="row items-center justify-evenly">
    <div>
      <h3>Welcome to GeoFlex!</h3>
      <p>This is your one-stop solution for conducting Geo experiments with Ads.</p>
      <p>
        The first and foremost prerequisite to get started is up get your business metrics
        attributed by geo unit and date. Once you get it, create a
        <router-link to="/datasources" class="text-primary" style="text-decoration: underline"
          >data source</router-link
        >
        and start exploring possible experiment designs in
        <router-link to="/experiments" class="text-primary" style="text-decoration: underline"
          >Experiments</router-link
        >.
      </p>
      <q-btn to="/configuration" label="Go to Configuration" color="primary" />
    </div>
  </q-page>
</template>

<script setup lang="ts">
//import { ref } from 'vue';

defineOptions({
  name: 'IndexPage',
});
</script>
